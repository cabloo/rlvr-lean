"""Draw the README's figures from the result summaries in docs/results/.

    python docs/figures/make_figures.py

Needs matplotlib. Every number drawn is read from the JSON files; nothing is typed in here.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
INK, MUTED, RULE = "#18211d", "#5a6962", "#dde3df"
RUNG_COLORS = {"below": "#1f6fb2", "in": "#7a4fb5", "above": "#c4551f"}
RUNG_NAMES = {"below": "hardest rung (base solves 7% of attempts)", "in": "middle rung (27%)", "above": "easy rung (77%)"}
SEED_COLORS = ["#1f6fb2", "#c4551f", "#7a4fb5"]
BASE_COLOR, TRAINED_COLOR = "#8a9690", "#0b6e5f"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(name):
    return json.loads((RESULTS / name).read_text())


def save(figure, name):
    figure.tight_layout()
    figure.savefig(HERE / name, format="svg")
    plt.close(figure)
    print("wrote", name)


def points(values):
    return [100 * value for value in values]


def rungs_by_round():
    data = load("ladder_l2_three_seeds.json")
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    rounds = [0, 1, 2, 3]
    for offset, rung in zip((-0.04, 0.0, 0.04), ("above", "in", "below")):
        triples = data["rungs"][rung]["minus_base"]
        mean = [0.0] + points(t[0] for t in triples)
        low = [0.0] + points(t[1] for t in triples)
        high = [0.0] + points(t[2] for t in triples)
        xs = [r + offset for r in rounds]
        axis.errorbar(xs, mean, yerr=[[m - lo for m, lo in zip(mean, low)], [hi - m for m, hi in zip(mean, high)]], color=RUNG_COLORS[rung],
                      marker="o", markersize=5, linewidth=2, capsize=3, label=RUNG_NAMES[rung])
    axis.axhline(0, color=MUTED, linewidth=1)
    axis.set_xticks(rounds, ["base", "1 round", "2 rounds", "3 rounds"])
    axis.set_ylabel("held-out pass rate, points over the base")
    axis.set_title("Every held-out rung rises; most of the gain comes in two rounds")
    axis.legend(loc="upper left", fontsize=9)
    save(figure, "rungs_by_round.svg")


def goal_set():
    data = load("ladder_l2_three_seeds.json")
    seeds = data["seeds"]
    solved = {
        "base, 32": sum(s["base"]["problems"] for s in seeds), "trained, 32": sum(s["m3"]["problems"] for s in seeds),
        "base, 93": sum(s["control"]["problems"] for s in seeds), "trained, 93": sum(s["equal_attempts"]["m3_problems"] for s in seeds),
    }
    rate = data["success_per_episode"]
    per_thousand = {"base, 32": 1000 * rate["base32"], "trained, 32": 1000 * rate["m3"], "base, 93": 1000 * rate["base93"], "trained, 93": 1000 * rate["m3_93"]}
    colors = [BASE_COLOR, TRAINED_COLOR, BASE_COLOR, TRAINED_COLOR]
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for axis, values, title, fmt in ((left, solved, "Goal problems solved", "{:.0f}"),
                                      (right, per_thousand, "Successes per 1,000 attempts", "{:.1f}")):
        bars = axis.bar(list(values), list(values.values()), color=colors, width=0.62)
        for bar, value in zip(bars, values.values()):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value), ha="center", va="bottom", fontsize=9)
        axis.set_title(title)
        axis.grid(axis="x", visible=False)
        axis.set_xticks(range(4), ["base\n32 attempts", "3 rounds\n32 attempts", "base\n93 attempts", "3 rounds\n93 attempts"], fontsize=8.5)
    left.set_ylabel("of 392 goal problems x 3 seeds")
    save(figure, "goal_set.svg")


def dose_curve():
    data = load("ladder_l1b_three_seeds.json")
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    per_pass = data["seeds"][0]["steps_per_pass"]
    for index, seed in enumerate(data["seeds"]):
        color = SEED_COLORS[index]
        left.plot([s / per_pass for s in seed["training_sample"]["step"]], seed["training_sample"]["body"], color=color, linewidth=1.2, alpha=0.55)
        left.plot([s / per_pass for s in seed["heldout"]["step"]], seed["heldout"]["body"], color=color, linewidth=2.2, label=f"seed {seed['seed']}")
    left.set_xlabel("passes over the same training proofs")
    left.set_ylabel("loss per proof token (nats)")
    left.set_title("Held-out loss (thick) turns up after one pass")
    left.set_ylim(bottom=0)
    left.legend(fontsize=9, loc="upper left", title="thin: training proofs", title_fontsize=8.5)
    for rung in ("above", "in", "below"):
        entry = data["pooled"][rung]
        mean, low, high = points(entry["minus_base"]), points(entry["low"]), points(entry["high"])
        right.errorbar(entry["pass"], mean, yerr=[[m - lo for m, lo in zip(mean, low)], [hi - m for m, hi in zip(mean, high)]],
                       color=RUNG_COLORS[rung], marker="o", markersize=4.5, linewidth=2, capsize=3, label=RUNG_NAMES[rung].split(" (")[0])
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xlabel("passes over the same training proofs")
    right.set_ylabel("pass rate, points over the base")
    right.set_title("...while held-out pass rate keeps rising")
    right.legend(fontsize=9, loc="upper left")
    save(figure, "dose_curve.svg")


def challenger():
    data = load("ladder_l2_three_seeds.json")
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    labels = [f"{b['round']}.{b['batch']}" for b in data["challenger"][0]["batches"]]
    xs = list(range(len(labels)))
    for index, seed in enumerate(data["challenger"]):
        left.plot(xs, points(b["share_known_false"] for b in seed["batches"]), color=SEED_COLORS[index], marker="o", markersize=4, linewidth=1.8, label=f"seed {seed['seed']}")
        right.plot(xs, [b["mean_pass_rate"] for b in seed["batches"]], color=SEED_COLORS[index], marker="o", markersize=4, linewidth=1.8)
    right.axhline(0.25, color=INK, linewidth=1.2, linestyle="--")
    right.text(xs[-1], 0.25, "target 0.25 ", ha="right", va="bottom", fontsize=9)
    for axis in (left, right):
        for boundary in (3.5, 7.5):
            axis.axvline(boundary, color=RULE, linewidth=1.2)
        axis.set_xticks(xs, labels, fontsize=8.5)
        axis.set_xlabel("round . batch")
    left.set_ylabel("% of the challenger's picks")
    left.set_ylim(0, 70)
    left.set_title("Refutations fall once it sees the new model")
    left.legend(fontsize=9, loc="lower left")
    right.set_ylabel("mean pass rate of the picks")
    right.set_ylim(0, 0.7)
    right.set_title("...but the picks get easier, not harder")
    save(figure, "challenger_picks.svg")


def pool_shape():
    data = load("candidates_by_predicted_rate.json")
    edges = data["edges"]
    centers = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    width = edges[1] - edges[0]
    everything = [t + f for t, f in zip(data["known_true"], data["known_false"])]
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    low, high = data["band"]
    axis.axvspan(low, high, color=TRAINED_COLOR, alpha=0.12, label="the band the reward aims at")
    axis.bar(centers, everything, width=width * 0.9, color=BASE_COLOR, label="all candidates")
    shown = [(c, f) for c, f in zip(centers, data["known_false"]) if f > 0]
    axis.plot([c for c, _ in shown], [f for _, f in shown], color=RUNG_COLORS["above"], marker="o", markersize=4, linewidth=1.8,
              label="of them, published false (solved by refuting)")
    axis.set_yscale("log")
    axis.set_ylim(1, 3e5)
    axis.set_xlim(0, 1)
    axis.set_xlabel("pass rate the challenger predicts for the base model")
    axis.set_ylabel("candidate problems (log scale)")
    axis.set_title(f"{data['candidates']:,} candidate problems: few in the middle")
    axis.legend(fontsize=8.5, loc="upper right")
    save(figure, "pool_shape.svg")


def lower_target():
    data = load("ladder_l2t_three_seeds.json")
    goal = data["goal_set_93_attempts"]
    arms = (("base", "base", BASE_COLOR), ("quarter_target", "3 rounds, target 1/4", RUNG_COLORS["in"]),
            ("low_target", "3 rounds, target 1/10", TRAINED_COLOR))
    seeds = list(range(data["seeds"]))
    width = 0.27
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for axis, values, title, fmt, room in ((left, goal["problems_solved_by_seed"], "Goal problems solved, 93 attempts each", "{:.0f}", 1.42),
                                            (right, goal["successes_per_1000_attempts_by_seed"], "Successes per 1,000 attempts", "{:.1f}", 1.14)):
        for offset, (key, label, color) in zip((-width, 0, width), arms):
            bars = axis.bar([seed + offset for seed in seeds], values[key], width=width * 0.92, color=color, label=label)
            for bar, value in zip(bars, values[key]):
                axis.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value), ha="center", va="bottom", fontsize=8)
        axis.set_ylim(0, max(max(values[key]) for key, _, _ in arms) * room)
        axis.set_xticks(seeds, [f"seed {seed}" for seed in seeds])
        axis.set_title(title)
        axis.grid(axis="x", visible=False)
    left.set_ylabel("of 392 goal problems")
    left.legend(fontsize=8.5, loc="upper left")
    save(figure, "lower_target.svg")


def repair():
    data = load("repair_checks.json")
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    attempts = ["1", "2", "3", "4", "5"]
    xs = list(range(len(attempts)))

    def total(by_attempt):
        return [100 * by_attempt[a] for a in attempts]

    def mark_lead(axis, upper, lead):
        for a in ("2", "5"):
            axis.annotate(f"+{100 * lead[a][0]:.1f}", (int(a) - 1, upper[int(a) - 1]), textcoords="offset points", xytext=(0, 7), ha="center",
                          fontsize=8.5, color=TRAINED_COLOR)

    first = data["first_check"]
    totals = first["hard_resolved_within"]
    left.plot(xs, total(totals["blind"]), color=BASE_COLOR, marker="o", markersize=5, linewidth=2, label="start over (blind)")
    left.plot(xs, total(totals["resume_with_state"]), color=TRAINED_COLOR, marker="o", markersize=5, linewidth=2, label="resume with Lean's proof state")
    left.plot(xs, total(totals["resume_without_state"]), color=TRAINED_COLOR, linestyle=":", marker="o", markersize=5, linewidth=2, label="resume without the state")
    left.fill_between(xs, total(totals["blind"]), total(totals["resume_with_state"]), color=TRAINED_COLOR, alpha=0.10, linewidth=0)
    mark_lead(left, total(totals["resume_with_state"]), first["hard_lead_within_resume_with_state_minus_blind"])
    left.set_title("Resuming after every failure")
    left.set_ylabel("% of hard episodes resolved so far")
    left.legend(fontsize=8.5, loc="upper left")

    second = data["second_check"]
    totals = second["hard_resolved_within"]
    right.plot(xs, total(totals["blind"]), color=BASE_COLOR, marker="o", markersize=5, linewidth=2, label="start over (blind)")
    right.plot(xs, total(totals["alternate"]), color=TRAINED_COLOR, marker="o", markersize=5, linewidth=2, label="repair once, then start over")
    right.fill_between(xs, total(totals["blind"]), total(totals["alternate"]), color=TRAINED_COLOR, alpha=0.10, linewidth=0)
    mark_lead(right, total(totals["alternate"]), second["hard_lead_within_alternate_minus_blind"])
    right.set_title("Repairing once, then starting over")
    right.legend(fontsize=8.5, loc="upper left")
    for axis in (left, right):
        axis.set_xticks(xs, ["1 attempt"] + [f"{a} attempts" for a in attempts[1:]], fontsize=8.5)
        axis.set_ylim(0, 100 * max(second["hard_resolved_within"]["alternate"].values()) * 1.3)
        axis.grid(axis="x", visible=False)
    save(figure, "repair.svg")


def accumulating_episode():
    data = load("accumulating_episode_three_seeds.json")
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    within = data["hard_resolved_within"]
    ks = sorted(within, key=int)
    xs = [int(k) for k in ks]
    episodes = data["hard_episodes"]
    blind = [100 * within[k]["blind"] / episodes for k in ks]
    keeps = [100 * within[k]["accumulate"] / episodes for k in ks]
    left.plot(xs, blind, color=BASE_COLOR, marker="o", markersize=5, linewidth=2, label="8 blind attempts")
    left.plot(xs, keeps, color=TRAINED_COLOR, marker="o", markersize=5, linewidth=2, label="an episode that keeps verified lemmas")
    left.fill_between(xs, blind, keeps, color=TRAINED_COLOR, alpha=0.10, linewidth=0)
    left.annotate(f"+{100 * data['primary'][0]:.1f}", (xs[-1], keeps[-1]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=TRAINED_COLOR)
    left.set_xticks(xs)
    left.set_xlabel("generations used on a problem")
    left.set_ylabel("% of hard episodes resolved so far")
    left.set_ylim(0, max(keeps) * 1.22)
    left.set_title("The lead grows with every generation")
    left.legend(fontsize=8.5, loc="upper left")
    left.grid(axis="x", visible=False)

    groups = data["goal_set_by_problem"]["by_proof_length"]
    names = list(groups)
    width = 0.38
    positions = list(range(len(names)))
    for offset, key, label, color in ((-width / 2, "blind", "8 blind attempts", BASE_COLOR), (width / 2, "accumulate", "keeps verified lemmas", TRAINED_COLOR)):
        heights = [groups[name][key] for name in names]
        bars = right.bar([p + offset for p in positions], heights, width=width * 0.94, color=color, label=label)
        for bar, height in zip(bars, heights):
            right.text(bar.get_x() + bar.get_width() / 2, height, f"{height}", ha="center", va="bottom", fontsize=8.5)
    right.set_xticks(positions, [f"{name} line" + ("" if name == "1" else "s") + f"\nof {groups[name]['problems']}" for name in names], fontsize=8.5)
    right.set_xlabel("shortest published proof; problems of that length")
    right.set_ylabel("never-solved problems solved")
    right.set_ylim(0, max(groups[name]["accumulate"] for name in names) * 1.25)
    right.set_title("More never-solved problems solved")
    right.legend(fontsize=8.5, loc="upper right")
    right.grid(axis="x", visible=False)
    save(figure, "accumulating_episode.svg")


if __name__ == "__main__":
    rungs_by_round()
    goal_set()
    dose_curve()
    challenger()
    pool_shape()
    lower_target()
    repair()
    accumulating_episode()
