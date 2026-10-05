"""The challenger of this stage CHOOSES pool problems; it does not write them. Spec: "The challenger in this stage".

It is a small predictor of a problem's pass rate under the current solver, and nothing else:

  what it reads    the base model's embedding of the statement (projected on its leading directions) and the
                   published facts the pool carries: where the problem is from, which side its certificate
                   proves, the source and the length of the published proof, STP's round number
  what it learns   every (problem, k, n) seen so far, recent rounds weighted most: a ridge-penalised binomial
                   logistic regression, the penalty chosen by cross-validation. It starts from L0's base map
  how it scores    by the EXPECTED reward of a candidate: k is a noisy count of the rate, and problems that
                   look alike differ, so k given the predicted rate is beta-binomial with one fitted dispersion
                   (0 is the plain binomial)
  how it chooses   the best `problems` of the candidates, with a share of the places given to random
                   candidates so that it keeps learning

The control arm is `random_draw`: the same candidates, a uniform seeded draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from rlvr_lean.domain.problem_pool.episodes import band, reward
from rlvr_lean.domain.problem_pool.selection import rank

SCORED, RANDOM_PLACE, RANDOM_DRAW = "scored", "random_place", "random_draw"      # how a proposal got its place
_SOURCES = ("goedel", "stp", "internlm_rows")           # internlm_proofs is the reference source
FACT_FEATURES = ("is_stp_conjecture", "is_known_false", *(f"source_{name}" for name in _SOURCES),
                 "log_published_proof_chars", "has_stp_round", "stp_round", "log_statement_chars")
_RATE_FLOOR = 1e-6


# ------------------------------------------------------------------------------------------------ features
def fact_features(rows: Sequence[Mapping]) -> np.ndarray:
    """One row of `FACT_FEATURES` per problem, from what the pool file and the published files say about it.
    Nothing here is measured by our model."""
    features = np.zeros((len(rows), len(FACT_FEATURES)))
    for index, row in enumerate(rows):
        stp_round = row.get("stp_round")
        features[index] = [
            row["kind"] == "stp_conjecture", row["side"] == "false",
            *(row.get("certificate_source") == name for name in _SOURCES),
            math.log1p(row.get("published_proof_chars") or 0),
            stp_round is not None, (stp_round or 0) / 50.0,
            math.log1p(len(row["statement"])),
        ]
    return features


@dataclass(frozen=True)
class Projection:
    """The leading directions of the statement embeddings (principal components), fitted once on every
    embedding the stage holds. No pass rate enters it."""
    mean: np.ndarray
    basis: np.ndarray            # (components, dimension)

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        return (np.asarray(embeddings, dtype=np.float64) - self.mean) @ self.basis.T


def fit_projection(embeddings: np.ndarray, components: int) -> Projection:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"a projection needs at least two embeddings in a 2-D array, got shape {values.shape}")
    mean = values.mean(axis=0)
    kept = max(1, min(components, values.shape[0] - 1, values.shape[1]))
    centered = values - mean
    if centered.shape[0] >= centered.shape[1]:
        # Many statements, few dimensions (24,000 embeddings of 4,096): the directions are the eigenvectors of the
        # dimension-by-dimension scatter matrix, which is far cheaper than a decomposition of the whole array.
        _, vectors = np.linalg.eigh(centered.T @ centered)
        basis = vectors[:, ::-1][:, :kept].T
    else:
        _, _, directions = np.linalg.svd(centered, full_matrices=False)
        basis = directions[:kept]
    return Projection(mean=mean, basis=np.ascontiguousarray(basis))


def feature_names(projection: Projection | int) -> tuple[str, ...]:
    """The names of the features `features_of` gives. `projection` may be the NUMBER of leading directions, where
    the projected statements are what is at hand (L2 stores them once, beside the embeddings)."""
    directions = projection if isinstance(projection, int) else projection.basis.shape[0]
    return (*(f"embedding_direction_{index}" for index in range(directions)), *FACT_FEATURES)


def features_of(rows: Sequence[Mapping], embeddings: np.ndarray, projection: Projection) -> np.ndarray:
    if len(rows) != len(embeddings):
        raise ValueError(f"{len(rows)} problems and {len(embeddings)} embeddings: one embedding per problem is needed")
    return np.hstack([projection.apply(embeddings), fact_features(rows)])


# ------------------------------------------------------------------------------------------- the predictor
@dataclass(frozen=True)
class PassRateModel:
    names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray     # one per (standardised) feature
    ridge: float
    dispersion: float = 0.0      # of k around the predicted rate: 0 is binomial

    def predict(self, features: np.ndarray) -> np.ndarray:
        standard = (np.asarray(features, dtype=np.float64) - self.center) / self.scale
        return _sigmoid(self.intercept + standard @ self.coefficients)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def recency_weights(rounds: Sequence[int], current_round: int, decay: float) -> np.ndarray:
    """The weight of an observation made in `rounds[i]` when round `current_round` is fitted: 1 for the current
    round, `decay` for the one before, `decay²` before that. Spec: "recent rounds weighted most"."""
    if not 0 < decay <= 1:
        raise ValueError(f"the recency decay must be in (0, 1], got {decay}")
    ages = current_round - np.asarray(rounds, dtype=np.float64)
    if np.any(ages < 0):
        raise ValueError("an observation is from a round later than the one being fitted")
    return decay ** ages


def _fit(standard: np.ndarray, resolved: np.ndarray, episodes: np.ndarray, weights: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
    """Newton's method on the ridge-penalised binomial log-likelihood (the intercept is not penalised)."""
    design = np.hstack([np.ones((standard.shape[0], 1)), standard])
    penalty = np.full(design.shape[1], float(ridge))
    penalty[0] = 0.0
    total = float(np.sum(weights * resolved))
    mean_rate = min(max(total / max(float(np.sum(weights * episodes)), 1e-12), _RATE_FLOOR), 1 - _RATE_FLOOR)
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(mean_rate / (1 - mean_rate))

    def loss(values: np.ndarray) -> float:
        rate = np.clip(_sigmoid(design @ values), _RATE_FLOOR, 1 - _RATE_FLOOR)
        likelihood = np.sum(weights * (resolved * np.log(rate) + (episodes - resolved) * np.log(1 - rate)))
        return float(-likelihood + 0.5 * np.sum(penalty * values ** 2))

    current = loss(beta)
    for _ in range(60):
        rate = _sigmoid(design @ beta)
        gradient = design.T @ (weights * (episodes * rate - resolved)) + penalty * beta
        curvature = (design * (weights * episodes * rate * (1 - rate))[:, None]).T @ design + np.diag(penalty + 1e-9)
        step = np.linalg.solve(curvature, gradient)
        scale = 1.0
        while scale > 1e-4 and loss(beta - scale * step) > current:
            scale /= 2
        if scale <= 1e-4:
            break
        beta = beta - scale * step
        improved, current = current, loss(beta)
        if improved - current < 1e-9 * max(1.0, abs(improved)):
            break
    return float(beta[0]), beta[1:]


def _standardise(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    return center, np.where(scale > 1e-9, scale, 1.0)       # a constant feature stays where it is and gets no weight


def _folds(problem_ids: Sequence[str], folds: int, seed: int) -> np.ndarray:
    order = sorted(range(len(problem_ids)), key=lambda index: rank(seed, "challenger_fold", problem_ids[index]))
    assignment = np.zeros(len(problem_ids), dtype=int)
    for position, index in enumerate(order):
        assignment[index] = position % folds
    return assignment


def binomial_deviance(rates: np.ndarray, resolved: np.ndarray, episodes: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Minus twice the binomial log-likelihood per episode: lower is a better prediction of k."""
    rate = np.clip(rates, _RATE_FLOOR, 1 - _RATE_FLOOR)
    weights = np.ones(len(rate)) if weights is None else weights
    likelihood = np.sum(weights * (resolved * np.log(rate) + (episodes - resolved) * np.log(1 - rate)))
    return float(-2 * likelihood / max(float(np.sum(weights * episodes)), 1e-12))


def fit_pass_rate_model(problem_ids: Sequence[str], features: np.ndarray, resolved: Sequence[int], episodes: Sequence[int],
                        names: Sequence[str], ridge_grid: Sequence[float], folds: int, dispersion_grid: Sequence[float],
                        seed: int, weights: Sequence[float] | None = None) -> tuple[PassRateModel, dict]:
    """The model fitted on every observation given, and how it was chosen: the ridge penalty with the lowest
    cross-validated deviance, and the dispersion with the highest beta-binomial likelihood of k around the
    OUT-OF-FOLD rates (rates the model gave problems it was not fitted on)."""
    features = np.asarray(features, dtype=np.float64)
    resolved, episodes = np.asarray(resolved, dtype=np.float64), np.asarray(episodes, dtype=np.float64)
    weights = np.ones(len(resolved)) if weights is None else np.asarray(weights, dtype=np.float64)
    if not (len(problem_ids) == len(features) == len(resolved) == len(episodes) == len(weights)) or len(resolved) < 2:
        raise ValueError("the challenger needs at least two observed problems, each with its features, k, n and weight")
    if np.any(resolved < 0) or np.any(resolved > episodes) or np.any(episodes <= 0):
        raise ValueError("every observation must be k of n with 0 <= k <= n and n > 0")
    folds = max(2, min(folds, len(resolved)))
    assignment = _folds(problem_ids, folds, seed)
    by_ridge, out_of_fold = {}, {}
    for ridge in ridge_grid:
        rates = np.zeros(len(resolved))
        for fold in range(folds):
            held, kept = assignment == fold, assignment != fold
            center, scale = _standardise(features[kept])
            intercept, coefficients = _fit((features[kept] - center) / scale, resolved[kept], episodes[kept], weights[kept], ridge)
            rates[held] = _sigmoid(intercept + ((features[held] - center) / scale) @ coefficients)
        by_ridge[float(ridge)] = binomial_deviance(rates, resolved, episodes, weights)
        out_of_fold[float(ridge)] = rates
    ridge = min(by_ridge, key=lambda value: (by_ridge[value], -value))     # a tie goes to the heavier penalty
    dispersion, by_dispersion = fit_dispersion(out_of_fold[ridge], resolved, episodes, dispersion_grid, weights)
    center, scale = _standardise(features)
    intercept, coefficients = _fit((features - center) / scale, resolved, episodes, weights, ridge)
    model = PassRateModel(tuple(names), center, scale, intercept, coefficients, ridge, dispersion)
    mean_rate = float(np.sum(weights * resolved) / np.sum(weights * episodes))
    chosen = {"observations": len(resolved), "folds": folds, "ridge": ridge, "deviance_by_ridge": {str(key): round(value, 5) for key, value in by_ridge.items()},
              "deviance_of_the_mean_rate": round(binomial_deviance(np.full(len(resolved), mean_rate), resolved, episodes, weights), 5),
              "dispersion": dispersion, "log_likelihood_by_dispersion": {str(key): round(value, 3) for key, value in by_dispersion.items()},
              "mean_rate": round(mean_rate, 5),
              "largest_coefficients": [[model.names[index], round(float(coefficients[index]), 4)]
                                       for index in np.argsort(-np.abs(coefficients))[:8]]}
    return model, {**chosen, "out_of_fold_rates": out_of_fold[ridge]}


# ---------------------------------------------------------------------------------- k around a predicted rate
def outcome_distribution(episodes: int, rate: float, dispersion: float = 0.0) -> np.ndarray:
    """P(k = 0..n) for a problem of predicted pass rate `rate`: binomial at dispersion 0, beta-binomial with
    that mean above it (the share of the variance that is between problems of one predicted rate)."""
    if episodes <= 0:
        raise ValueError(f"the number of episodes must be positive, got {episodes}")
    if not 0 <= dispersion < 1:
        raise ValueError(f"the dispersion must be in [0, 1), got {dispersion}")
    rate = min(max(float(rate), _RATE_FLOOR), 1 - _RATE_FLOOR)
    counts = range(episodes + 1)
    if dispersion == 0:
        return np.array([math.comb(episodes, k) * rate ** k * (1 - rate) ** (episodes - k) for k in counts])
    size = 1 / dispersion - 1
    alpha, beta = rate * size, (1 - rate) * size
    log_pmf = [math.lgamma(episodes + 1) - math.lgamma(k + 1) - math.lgamma(episodes - k + 1)
               + math.lgamma(k + alpha) + math.lgamma(episodes - k + beta) - math.lgamma(episodes + alpha + beta)
               - math.lgamma(alpha) - math.lgamma(beta) + math.lgamma(alpha + beta) for k in counts]
    return np.exp(np.array(log_pmf))


def fit_dispersion(rates: np.ndarray, resolved: np.ndarray, episodes: np.ndarray, grid: Sequence[float],
                   weights: np.ndarray | None = None) -> tuple[float, dict[float, float]]:
    """The dispersion of the grid under which the observed k are most likely given `rates`."""
    weights = np.ones(len(rates)) if weights is None else weights
    likelihoods = {}
    for dispersion in grid:
        total = 0.0
        for rate, k, n, weight in zip(rates, resolved, episodes, weights):
            total += weight * math.log(max(float(outcome_distribution(int(n), float(rate), float(dispersion))[int(k)]), 1e-300))
        likelihoods[float(dispersion)] = total
    best = max(likelihoods, key=lambda value: (likelihoods[value], -value))
    return best, likelihoods


def expected_reward(rate: float, episodes: int, target_rate: float, dispersion: float = 0.0) -> float:
    """What a problem of predicted pass rate `rate` is expected to earn the challenger over n episodes."""
    distribution = outcome_distribution(episodes, rate, dispersion)
    return float(sum(distribution[k] * reward(k, episodes, target_rate) for k in range(episodes + 1)))


def expected_rewards(rates: Sequence[float], episodes: int, target_rate: float, dispersion: float = 0.0) -> np.ndarray:
    return np.array([expected_reward(rate, episodes, target_rate, dispersion) for rate in rates])


# ------------------------------------------------------------------------------------------------- choosing
def propose(candidate_ids: Sequence[str], scores: Sequence[float], problems: int, random_share: float, seed: int) -> list[dict]:
    """The challenger's round: the best-scoring candidates, with `random_share` of the places given to a seeded
    random draw of the others. Ties go by the seeded order. Fewer candidates than places: all of them."""
    if len(candidate_ids) != len(scores):
        raise ValueError("one score per candidate is needed")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("a candidate appears twice: a problem is proposed once")
    if not 0 <= random_share < 1:
        raise ValueError(f"the share of random places must be in [0, 1), got {random_share}")
    places = min(problems, len(candidate_ids))
    random_places = int(round(places * random_share))
    by_score = sorted(range(len(candidate_ids)), key=lambda index: (-float(scores[index]), rank(seed, "propose", candidate_ids[index])))
    chosen = [{"problem_id": candidate_ids[index], "how": SCORED, "score": round(float(scores[index]), 6)}
              for index in by_score[:places - random_places]]
    rest = sorted(by_score[places - random_places:], key=lambda index: rank(seed, "random_place", candidate_ids[index]))
    chosen += [{"problem_id": candidate_ids[index], "how": RANDOM_PLACE, "score": round(float(scores[index]), 6)}
               for index in rest[:random_places]]
    return chosen


def random_draw(candidate_ids: Sequence[str], problems: int, seed: int, scores: Sequence[float] | None = None) -> list[dict]:
    """The control arm's round: a uniform seeded draw of the same candidates. The challenger's score of each is
    recorded beside it (it chose nothing here), so the two arms can be compared on what the challenger expected."""
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("a candidate appears twice: a problem is proposed once")
    order = sorted(range(len(candidate_ids)), key=lambda index: rank(seed, "random_draw", candidate_ids[index]))
    return [{"problem_id": candidate_ids[index], "how": RANDOM_DRAW,
             "score": None if scores is None else round(float(scores[index]), 6)} for index in order[:min(problems, len(candidate_ids))]]


# ---------------------------------------------------------------------------------------------- calibration
CALIBRATION_EDGES = (0.0, 0.05, 0.13, 0.25, 0.41, 0.6, 0.8, 1.0)


def calibration_bins(rates: np.ndarray, resolved: np.ndarray, episodes: np.ndarray, edges: Sequence[float] = CALIBRATION_EDGES) -> list[dict]:
    """The observed pass rate by predicted-rate bin (empty bins are left out)."""
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (rates >= low) & ((rates < high) | (high == edges[-1]))
        if inside.any():
            bins.append({"predicted_from": low, "predicted_to": high, "problems": int(inside.sum()),
                         "mean_predicted": round(float(rates[inside].mean()), 4),
                         "observed_rate": round(float(resolved[inside].sum() / episodes[inside].sum()), 4)})
    return bins


def calibration(rates: Sequence[float], resolved: Sequence[int], episodes: Sequence[int], target_rate: float, floor: float,
                dispersion: float, edges: Sequence[float] = CALIBRATION_EDGES) -> dict:
    """How the predicted rates read against what was measured, on problems the model was NOT fitted on: the
    observed rate by predicted-rate bin, the deviance against the one-rate-for-all prediction, and what the
    challenger would have earned by taking the best tenth by expected reward."""
    rates = np.asarray(rates, dtype=np.float64)
    resolved, episodes = np.asarray(resolved, dtype=np.float64), np.asarray(episodes, dtype=np.float64)
    if len(rates) == 0:
        return {"problems": 0}
    bins = calibration_bins(rates, resolved, episodes, edges)
    earned = np.array([reward(int(k), int(n), target_rate) for k, n in zip(resolved, episodes)])
    expected = np.array([expected_reward(float(rate), int(n), target_rate, dispersion) for rate, n in zip(rates, episodes)])
    top = np.argsort(-expected)[:max(1, len(rates) // 10)]
    low, high = band(target_rate, floor)
    mean_rate = float(resolved.sum() / episodes.sum())
    return {"problems": len(rates), "bins": bins,
            "deviance": round(binomial_deviance(rates, resolved, episodes), 5),
            "deviance_of_the_mean_rate": round(binomial_deviance(np.full(len(rates), mean_rate), resolved, episodes), 5),
            "mean_reward_of_all": round(float(earned.mean()), 4),
            "mean_reward_of_the_best_tenth_by_expected_reward": round(float(earned[top].mean()), 4),
            "share_in_the_band_of_all": round(float(np.mean((resolved / episodes >= low) & (resolved / episodes <= high))), 4),
            "share_in_the_band_of_the_best_tenth": round(float(np.mean((resolved[top] / episodes[top] >= low) & (resolved[top] / episodes[top] <= high))), 4)}
