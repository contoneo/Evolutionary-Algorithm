import warnings

import numpy as np
import pandas as pd

# For every matrix containing samples in the way (t_start, t_end) the columns are numerated:
t_start_column = 0
t_end_column = 1

# T_Start: Instant when pre-crash recording time starts.
T_Start = -5000
# t_0: Instant when pre-crash recording time ends.
# As t_0 is the time when the recording algorithm starts, it is normally set to zero.
t_0 = 0
# Possible offset of the range (T_Start, t_0) for the creation of samples in step_1
offset = 10

# popsize: size of a test set (EA's output, run on the HiL as one batch)
popsize = 4
# lambda_new: number of candidates generated each cycle, before the greedy test-set selection
# (also the number of selected parents plus immigrants)
lambda_new = 20
# max_attempts: retry rounds for candidate generation, so a cycle that only produces duplicates
# does not loop forever
max_attempts = 20

# --- Fitness weights (all terms are normalised by |T|, so they are unitless) ---
# w: weight of t_end vs t_start in V1 and in the whole-tuple distance d. w > 0.5 means
# "t_end matters more than t_start".
w = 0.75
# w_explore: the same weight in the distance used for coverage decisions (the farthest-first
# picks of the exploration slots and the tie-break of the test-set selection). 0.5 makes a gap
# in t_start count as much as the same gap in t_end, so the domain is covered evenly in ms.
w_explore = 0.5
# gamma: weight of V2 (the out-of-T penalty) against V1.
gamma = 10
# r: novelty radius (normalised by |T|) for V3.
r = 0.01
# beta: weight of V3 (the novelty/Pesticide-Paradox term). The optimum sits on a ring of
# radius r around a tested tuple only when beta > r.
beta = 0.05

# s: selection pressure for linear-ranking roulette, s in [1, 2].
s = 1.5

# --- Mutation ---
# sigma: standard deviation of the normal mutation (ms), adapted per cycle (see update_sigma).
sigma = 50
sigma_min = 1
# sigma_max_ratio: upper bound of sigma as a share of |T|, so it follows any change of T.
sigma_max_ratio = 0.1
# patience: cycles without a new failure before sigma grows (explore further).
patience = 2
# immigrant_rate: fraction of each candidate batch drawn fresh from the run's HaltonSampler
# instead of from mutation, to keep exploring (Pesticide Paradox).
immigrant_rate = 0.2
# explore_slots: test-set slots reserved for pure exploration (see select_explorers). Without
# them, once E is non-empty V1 pulls every slot toward the known errors, and failure regions
# far from them are practically never found.
explore_slots = 1
# explore_every: k. Once E is non-empty, every k-th cycle is a coverage cycle: the whole test set
# explores, as while E is empty, so failure regions far from the known ones can still be found;
# the other cycles exploit E. 0 turns coverage cycles off.
explore_every = 3
# explore_pool: size of the fresh pool the exploration slots are picked from (farthest first).
# It must hold at least popsize tuples, since while E is empty every slot explores (coverage
# mode); a larger pool gives the farthest-point choice more to choose from, so the tested tuples
# cover the domain more evenly.
explore_pool = 100
# d_min: minimum event duration (t_end - t_start), so a mutated tuple is never zero-length.
d_min = 1

# --- Boundary candidates ---
# resolution_ms: boundary resolution; no midpoint is proposed between an error and its nearest
# pass once they are closer than this (weighted distance, see _weighted_distance_ms).
resolution_ms = 1

# --- Reproducibility ---
# seed: last seed used to create `rng`. `rng` is the single np.random.Generator that every
# random function below is given; reseed it with seed_rng(seed) for a reproducible run.
seed = None
rng = np.random.default_rng(seed)


def seed_rng(new_seed):
    global seed, rng
    seed = new_seed
    rng = np.random.default_rng(seed)
    return rng


# error_hit_set: tuples manually known to have found an error, merged into a fresh archive
# as fail rows with cycle = 0 (see merge_manual_error_hit_set).
error_hit_set = np.array([])


def _abs_T():
    return abs(t_0 - T_Start)


def check_valid_parameters():
    """Checks across parameters that a single input's limits can't express. Returns
    (errors, cautions): an error makes a run impossible or its test sets short, so no run may
    start; a caution only weakens the search."""
    errors, cautions = [], []
    if T_Start >= t_0:
        errors.append(f"T_Start ({T_Start}) must be before t_0 ({t_0}): T = [T_Start, t_0] is empty.")
    if lambda_new < popsize - explore_slots:
        errors.append(
            f"lambda_new ({lambda_new}) must be at least popsize - explore_slots "
            f"({popsize} - {explore_slots} = {popsize - explore_slots}): the greedy test-set "
            "selection fills those slots from the lambda_new candidates, so fewer candidates "
            "would make the test set shorter than popsize."
        )
    if explore_pool < popsize:
        errors.append(
            f"explore_pool ({explore_pool}) must be at least popsize ({popsize}): while E is "
            "empty every slot is an exploration slot, picked from these tuples, so a smaller "
            "pool would make the test set shorter than popsize."
        )
    if T_Start < t_0 and sigma_min > sigma_max_ratio * _abs_T():
        errors.append(
            f"sigma_min ({sigma_min}) must not exceed the upper bound of sigma, "
            f"sigma_max_ratio * |T| = {sigma_max_ratio} * {_abs_T()} = {sigma_max_ratio * _abs_T():g} ms."
        )
    if explore_every == 1:
        cautions.append(
            "explore_every = 1 makes every cycle a coverage cycle: the known failures are never "
            "exploited. Use 0 (off) or 2 and above."
        )
    if beta <= r:
        cautions.append(
            f"beta ({beta}) should be greater than r ({r}): otherwise the optimum of f doesn't "
            "sit on a ring around a known error, and V1 and V3 no longer balance near it."
        )
    return errors, cautions


def _repair(samples):
    """Bring back into the valid domain [T_Start-offset, t_0+offset], swap t_start/t_end if
    inverted, then enforce the minimum event duration d_min. An overshoot is mirrored back
    inside at the bound it crossed (a reflecting bound), so mutants don't pile up exactly on the
    bound as clipping would make them; an overshoot wider than the whole domain is clipped. The
    domain bound keeps mutation from drifting candidates away from the test case indefinitely
    during pure novelty search (V1 = 0, nothing else pulls t_start back); t_start may still lie
    before T_Start, as long as it stays within `offset` of it."""
    samples = np.asarray(samples, dtype=int).reshape(-1, 2).copy()
    if samples.shape[0] == 0:
        return samples
    lo, hi = T_Start - offset, t_0 + offset
    samples = np.where(samples < lo, 2 * lo - samples, samples)
    samples = np.where(samples > hi, 2 * hi - samples, samples)
    samples = np.clip(samples, lo, hi)
    swap = samples[:, t_start_column] > samples[:, t_end_column]
    samples[swap] = samples[swap][:, ::-1]
    too_short = (samples[:, t_end_column] - samples[:, t_start_column]) < d_min
    samples[too_short, t_end_column] = samples[too_short, t_start_column] + d_min
    # Stretching may push t_end past hi: then shift the tuple back so it ends at hi.
    beyond = samples[:, t_end_column] > hi
    samples[beyond, t_end_column] = hi
    samples[beyond, t_start_column] = hi - d_min
    return samples


def initialize_samples_array(n=None, rng=None):
    """Uniformly random test samples: n tuples drawn in the pre-crash recording time T extended
    by `offset` on both sides; duration is repaired to at least d_min. The EA itself draws its
    fresh points from a HaltonSampler (more even); these random ones serve as a baseline."""
    if n is None:
        n = popsize
    if rng is None:
        rng = globals()["rng"]
    lo, hi = T_Start - offset, t_0 + offset
    samples = rng.uniform(lo, hi, size=(n, 2))
    samples = np.rint(samples).astype(int)
    return _repair(samples)


def _radical_inverse(index, base):
    """Van der Corput radical inverse of every integer in index, in the given base."""
    index = np.asarray(index, dtype=np.int64).copy()
    result = np.zeros(index.shape, dtype=float)
    fraction = 1.0 / base
    while np.any(index > 0):
        result += fraction * (index % base)
        index //= base
        fraction /= base
    return result


class HaltonSampler:
    """Low-discrepancy points over the search domain, for every fresh point of a run (Step 1,
    immigrants, exploration pool): the 2-D Halton sequence (bases 2 and 3), shifted at random
    once per run (Cranley-Patterson; the shift comes from the run's generator, so a seed
    reproduces it) and continued across cycles, so later points fill the gaps between earlier
    ones. t_start may lie before T_Start (only t_end is penalised, by V2)."""

    def __init__(self, rng=None):
        if rng is None:
            rng = globals()["rng"]
        self.shift = rng.random(2)
        self.index = 1  # index 0 is the origin of the unshifted sequence

    def unit_points(self, n):
        """The next n points of the shifted sequence, in the unit square."""
        index = np.arange(self.index, self.index + n)
        self.index += n
        points = np.stack((_radical_inverse(index, 2), _radical_inverse(index, 3)), axis=1)
        return (points + self.shift) % 1.0

    def draw(self, n):
        """The next n points as tuples of the domain: an area-preserving map of the unit square
        onto the triangle t_start <= t_end of [T_Start-offset, t_0+offset] (t_end = lo + span *
        sqrt(u), t_start uniform in [lo, t_end]), rounded and repaired (d_min)."""
        u = self.unit_points(n)
        lo, hi = T_Start - offset, t_0 + offset
        t_end = lo + (hi - lo) * np.sqrt(u[:, 0])
        t_start = lo + (t_end - lo) * u[:, 1]
        samples = np.zeros((n, 2))
        samples[:, t_start_column], samples[:, t_end_column] = t_start, t_end
        return _repair(np.rint(samples).astype(int))


def _weighted_distance_ms(a, b, weight=None):
    """Pairwise weighted distance in ms between every row of a and every row of b.
    Shape (len(a), len(b)). The weight (default w) favours t_end over t_start."""
    if weight is None:
        weight = w
    a = np.asarray(a).reshape(-1, 2)
    b = np.asarray(b).reshape(-1, 2)
    diff = np.abs(a[:, None, :] - b[None, :, :])
    return (1 - weight) * diff[:, :, t_start_column] + weight * diff[:, :, t_end_column]


def distance_to_set(samples, ref, weight=None):
    """Normalised whole-tuple distance d(x, y) from each sample to the nearest row in ref, with
    the given weight (default w; coverage decisions use w_explore)."""
    samples = np.asarray(samples).reshape(-1, 2)
    ref = np.asarray(ref).reshape(-1, 2)
    if ref.size == 0:
        return np.full(samples.shape[0], np.inf)
    return _weighted_distance_ms(samples, ref, weight).min(axis=1) / _abs_T()


def V_1(samples_array, error_set):
    """t_start and t_end are independent coordinates: the nearest E_i[t_start] and the nearest
    E_j[t_end] are taken separately, so they may come from different tuples of error_set
    (see README "Fitness Orientation"). 0 if error_set is empty."""
    samples_array = np.asarray(samples_array).reshape(-1, 2)
    error_set = np.asarray(error_set).reshape(-1, 2)
    if error_set.size == 0:
        return np.zeros(samples_array.shape[0])

    diff = np.abs(samples_array[:, None, :] - error_set[None, :, :])
    nearest = diff.min(axis=1)  # shape (n, 2): per-coordinate minimum distance
    weighted = (1 - w) * nearest[:, t_start_column] + w * nearest[:, t_end_column]
    return weighted / _abs_T()


def V_2(samples_array):
    """Soft penalty for t_end lying outside T = [T_Start, t_0], normalised by |T|."""
    samples_array = np.asarray(samples_array).reshape(-1, 2)
    t_end = samples_array[:, t_end_column]
    distance = np.maximum(T_Start - t_end, t_end - t_0)
    return distance.clip(min=0) / _abs_T()


def V_3(samples_array, tested_set):
    """Novelty term (Pesticide Paradox): repels samples from everything already tested (A).
    Whole-tuple distance, so it never forbids reusing a single known coordinate."""
    samples_array = np.asarray(samples_array).reshape(-1, 2)
    tested_set = np.asarray(tested_set).reshape(-1, 2)
    if tested_set.size == 0:
        return np.zeros(samples_array.shape[0])
    d = distance_to_set(samples_array, tested_set)
    return np.maximum(0, 1 - d / r)


def fitness_components(samples, error_set, tested_set):
    return V_1(samples, error_set), V_2(samples), V_3(samples, tested_set)


def fitness_function(samples, error_set, tested_set):
    v1, v2, v3 = fitness_components(samples, error_set, tested_set)
    return v1 + gamma * v2 + beta * v3


def rank_probabilities(scores, pressure=None):
    """Linear-ranking roulette: rank 0 is the worst (highest f), rank n-1 the best (lowest f).
    p = (2-s)/n + 2*rank*(s-1)/(n*(n-1)), s in [1, 2]. Scale-free, safe for f = 0."""
    if pressure is None:
        pressure = s
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 1:
        return np.array([1.0])
    order = np.argsort(-scores)  # order[0]: index of the worst (highest f)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    return (2 - pressure) / n + 2 * ranks * (pressure - 1) / (n * (n - 1))


def select_parents(probabilities, pool, n, rng=None):
    """Roulette wheel: n spins, each picks the pool tuple k with
    sum(p_1..p_k-1) <= rand < sum(p_1..p_k). A tuple may be picked more than once."""
    if rng is None:
        rng = globals()["rng"]
    x = rng.random(n)
    cumulative = np.cumsum(probabilities)
    selection = np.digitize(x, cumulative)
    selection = np.clip(selection, 0, pool.shape[0] - 1)
    return pool[selection, :]


def update_sigma(sigma_value, new_failures, cycles_without_failure):
    """Adaptive step size driven by HiL feedback: shrink to exploit a fresh failure, grow after
    `patience` quiet cycles to explore again. Clipped to [sigma_min, sigma_max_ratio * |T|]."""
    if new_failures:
        sigma_value = sigma_value * 0.7
        cycles_without_failure = 0
    else:
        cycles_without_failure += 1
        if cycles_without_failure >= patience:
            sigma_value = sigma_value * 1.5
            cycles_without_failure = 0
    sigma_value = float(np.clip(sigma_value, sigma_min, sigma_max_ratio * _abs_T()))
    return sigma_value, cycles_without_failure


def n_immigrants_per_batch():
    """Share of the lambda_new candidates drawn fresh instead of mutated; the rest are mutants."""
    return int(round(lambda_new * immigrant_rate))


def generate_candidates(parents, tested, sigma_value, n_immigrants=None, rng=None, sampler=None):
    """Normal mutation of each selected parent once, plus n_immigrants random immigrants,
    repaired and de-duplicated against each other and against everything already tested.
    Retries up to max_attempts rounds (each round mutates every parent again, with fresh
    noise) until len(parents) + n_immigrants new candidates exist, so a batch that only
    produces duplicates does not stall the cycle."""
    if n_immigrants is None:
        n_immigrants = n_immigrants_per_batch()
    if rng is None:
        rng = globals()["rng"]
    if sampler is None:
        sampler = HaltonSampler(rng)
    tested = np.asarray(tested).reshape(-1, 2)
    parents = np.asarray(parents).reshape(-1, 2)
    n = parents.shape[0] + n_immigrants

    collected = np.empty((0, 2), dtype=int)
    for _ in range(max_attempts):
        mutation = np.rint(rng.normal(0, sigma_value, size=parents.shape)).astype(int)
        mutants = parents + mutation

        immigrants = (
            sampler.draw(n_immigrants)
            if n_immigrants > 0
            else np.empty((0, 2), dtype=int)
        )

        batch = _repair(np.concatenate((mutants, immigrants), axis=0))
        collected = np.unique(np.concatenate((collected, batch), axis=0), axis=0)

        if tested.size:
            already_tested = (collected[:, None, :] == tested[None, :, :]).all(axis=2).any(axis=1)
            collected = collected[~already_tested]

        if collected.shape[0] >= n:
            break

    return collected


def select_test_set(candidates, error_set, tested_set, n=None):
    """Greedy set selection (comma selection: tested tuples never return). Repeat n times: score
    every remaining candidate against A u chosen, take the lowest, add it. V3 then also
    spaces the chosen tuples apart, so the set spreads over the error clusters. Ties go to the
    candidate farthest (by d) from A u chosen (farthest first)."""
    if n is None:
        n = popsize
    candidates = np.asarray(candidates).reshape(-1, 2)
    tested_set = np.asarray(tested_set).reshape(-1, 2)

    chosen = np.empty((0, 2), dtype=int)
    remaining = candidates.copy()
    rounds = min(n, remaining.shape[0])
    for _ in range(rounds):
        archive_so_far = np.concatenate((tested_set, chosen), axis=0)
        scores = fitness_function(remaining, error_set, archive_so_far)
        # Farthest first among equal scores. While E is empty every candidate farther than r
        # from A u chosen scores exactly 0; taking the first of them would always take the
        # earliest t_start (the candidates are sorted), piling the search up at the left edge.
        tied = np.flatnonzero(np.isclose(scores, scores.min(), rtol=0.0, atol=1e-12))
        best_idx = int(
            tied[np.argmax(distance_to_set(remaining[tied], archive_so_far, weight=w_explore))]
        )
        chosen = np.concatenate((chosen, remaining[best_idx : best_idx + 1]), axis=0)
        remaining = np.delete(remaining, best_idx, axis=0)

    if chosen.shape[0] < n:
        warnings.warn(f"Only {chosen.shape[0]} candidates available; requested {n}.")
    return chosen


def select_explorers(pool, tested_set, n=None):
    """Pure exploration, no V1: greedy farthest-point selection. Repeat n times: take the pool
    tuple farthest from A u chosen (by d with weight w_explore), with V2 still keeping t_end
    inside T. The pool is only a sample, so each pick approximates the largest untested gap."""
    if n is None:
        n = explore_slots
    pool = np.asarray(pool).reshape(-1, 2)
    tested_set = np.asarray(tested_set).reshape(-1, 2)

    chosen = np.empty((0, 2), dtype=int)
    remaining = pool.copy()
    for _ in range(min(n, remaining.shape[0])):
        reference = np.concatenate((tested_set, chosen), axis=0)
        scores = gamma * V_2(remaining) - distance_to_set(remaining, reference, weight=w_explore)
        best_idx = int(np.argmin(scores))
        chosen = np.concatenate((chosen, remaining[best_idx : best_idx + 1]), axis=0)
        remaining = np.delete(remaining, best_idx, axis=0)
    return chosen


def boundary_candidates(error_set, passed_set, resolution=None):
    """For each error e, take the nearest passed tuple p. If d(e, p) is wider than `resolution`,
    propose the rounded midpoint as a candidate: it competes like any other candidate, and
    passed tuples repel through V3, so the search fills in the boundary region."""
    if resolution is None:
        resolution = resolution_ms
    error_set = np.asarray(error_set).reshape(-1, 2)
    passed_set = np.asarray(passed_set).reshape(-1, 2)
    if error_set.size == 0 or passed_set.size == 0:
        return np.empty((0, 2), dtype=int)

    weighted = _weighted_distance_ms(error_set, passed_set)
    nearest_idx = np.argmin(weighted, axis=1)
    nearest_dist = weighted[np.arange(error_set.shape[0]), nearest_idx]

    wide_pairs = nearest_dist > resolution
    if not wide_pairs.any():
        return np.empty((0, 2), dtype=int)

    midpoints = np.rint((error_set[wide_pairs] + passed_set[nearest_idx[wide_pairs]]) / 2)
    return np.unique(_repair(midpoints), axis=0)


# --- Archive: a DataFrame of every tuple run on the HiL so far, with its verdict and cycle, ---
# plus the failures added by hand. source: "hil" for a tuple run on the HiL, "manual" for a
# failure entered by hand, "override" for a tuple that passed on the HiL in an earlier cycle and
# was then marked as failed by hand (see merge_manual_error_hit_set).
ARCHIVE_COLUMNS = ["t_start", "t_end", "verdict", "cycle", "source"]


def new_archive():
    return pd.DataFrame(
        {
            name: pd.Series(dtype=str if name == "source" else int)
            for name in ARCHIVE_COLUMNS
        }
    )


def record_results(archive, test_set, verdicts, cycle):
    test_set = np.asarray(test_set).reshape(-1, 2)
    verdicts = np.asarray(verdicts).astype(int)
    rows = pd.DataFrame(
        {
            "t_start": test_set[:, t_start_column],
            "t_end": test_set[:, t_end_column],
            "verdict": verdicts,
            "cycle": cycle,
            "source": "hil",
        }
    )
    return pd.concat([archive, rows], ignore_index=True)


def archive_error_set(archive):
    if archive.empty:
        return np.empty((0, 2), dtype=int)
    fails = archive[archive["verdict"] == 1]
    return fails[["t_start", "t_end"]].to_numpy(dtype=int)


def archive_passed_set(archive):
    if archive.empty:
        return np.empty((0, 2), dtype=int)
    passed = archive[archive["verdict"] == 0]
    return passed[["t_start", "t_end"]].to_numpy(dtype=int)


def archive_tested_set(archive):
    if archive.empty:
        return np.empty((0, 2), dtype=int)
    return archive[["t_start", "t_end"]].to_numpy(dtype=int)


def merge_manual_error_hit_set(archive, manual_error_hit_set, cycle=0):
    """Add manually known failures to E at `cycle`. A tuple not in the archive yet becomes a
    fail row with source "manual". A pass of the current test set (source "hil" at `cycle`) is
    marked as failed, like checking its box. A pass from an earlier cycle is marked as failed
    with source "override" and moved to `cycle`, so it counts as a new failure of this cycle.
    A tuple already in E stays as it is. Failures are only ever added, never removed."""
    manual = np.asarray(manual_error_hit_set, dtype=int).reshape(-1, 2)
    if manual.size == 0:
        return archive
    manual = np.unique(manual, axis=0)
    archive = archive.copy()
    row_of = {
        (t_start, t_end): label
        for label, t_start, t_end in zip(archive.index, archive["t_start"], archive["t_end"])
    }
    new = []
    for t_start, t_end in manual:
        label = row_of.get((t_start, t_end))
        if label is None:
            new.append((t_start, t_end))
        elif archive.loc[label, "verdict"] == 0:
            archive.loc[label, "verdict"] = 1
            if archive.loc[label, "cycle"] != cycle:
                archive.loc[label, ["cycle", "source"]] = [cycle, "override"]
    if not new:
        return archive
    new = np.array(new, dtype=int)
    rows = pd.DataFrame(
        {
            "t_start": new[:, t_start_column],
            "t_end": new[:, t_end_column],
            "verdict": 1,
            "cycle": cycle,
            "source": "manual",
        }
    )
    return pd.concat([archive, rows], ignore_index=True)


def parse_error_hit_set(text):
    """Parse a manually entered error_hit_set, one "t_start, t_end" tuple per line (blank lines
    are skipped). Every line must hold exactly two integers, last at least d_min long and lie
    in the domain [T_Start-offset, t_0+offset]; otherwise a ValueError lists the bad lines."""
    lo, hi = T_Start - offset, t_0 + offset
    tuples, problems = [], []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values = [int(value) for value in line.split(",")]
        except ValueError:
            problems.append(f"line {number}: not a whole number")
            continue
        if len(values) != 2:
            problems.append(f"line {number}: expected 2 values (t_start, t_end), got {len(values)}")
            continue
        t_start, t_end = values
        if t_end - t_start < d_min:
            problems.append(f"line {number}: t_end - t_start must be at least d_min = {d_min}")
        elif not (lo <= t_start <= hi and lo <= t_end <= hi):
            problems.append(f"line {number}: outside the domain [{lo}, {hi}]")
        else:
            tuples.append(values)
    if problems:
        raise ValueError("; ".join(problems))
    if not tuples:
        return np.empty((0, 2), dtype=int)
    return np.unique(np.array(tuples, dtype=int), axis=0)


def step_1(rng=None, sampler=None):
    """README "Algorithm", item 1. Step 1: the first test set, popsize distinct tuples, the
    first points of the run's HaltonSampler. Duplicates are redrawn (up to max_attempts rounds),
    keeping the order of the draws, so the HiL never runs a tuple twice in the first set either."""
    if sampler is None:
        sampler = HaltonSampler(rng)
    samples = np.empty((0, 2), dtype=int)
    for _ in range(max_attempts):
        samples = np.concatenate((samples, sampler.draw(popsize - samples.shape[0])), axis=0)
        _, first = np.unique(samples, axis=0, return_index=True)
        samples = samples[np.sort(first)]
        if samples.shape[0] == popsize:
            break
    return samples


def _exploit(error_set, passed_set, tested_set, last_test_set, sigma_value, n_exploit, rng, sampler):
    """README "Algorithm", items 5.1-5.4: the n_exploit slots that exploit E (E is not empty)."""
    # 5.1 Parent selection. The pool is the last test set u E, so every error cluster stays
    # alive even when it wasn't part of the last test set; linear-ranking roulette selects one
    # parent per mutant (lambda_new minus the immigrants).
    parent_pool = np.unique(
        np.concatenate((np.asarray(last_test_set).reshape(-1, 2), error_set), axis=0), axis=0
    )
    n_immigrants = n_immigrants_per_batch()
    scores = fitness_function(parent_pool, error_set, tested_set)
    p = rank_probabilities(scores)
    selected_parents = select_parents(p, parent_pool, lambda_new - n_immigrants, rng=rng)

    # 5.2 Mutation: each selected parent mutated once, plus random immigrants.
    candidates = generate_candidates(
        selected_parents, tested_set, sigma_value, n_immigrants=n_immigrants, rng=rng,
        sampler=sampler,
    )

    # 5.3 Boundary candidates: midpoints between each error and its nearest pass.
    extra = boundary_candidates(error_set, passed_set)
    if extra.size:
        if tested_set.size:
            already_tested = (extra[:, None, :] == tested_set[None, :, :]).all(axis=2).any(axis=1)
            extra = extra[~already_tested]
        candidates = np.unique(np.concatenate((candidates, extra), axis=0), axis=0)

    # 5.4 Test set selection: greedy, from the candidates only.
    return select_test_set(candidates, error_set, tested_set, n=n_exploit)


def is_coverage_cycle(error_set, cycle):
    """Whether the cycle numbered `cycle` is a coverage cycle, where the whole test set explores:
    while E is empty, and from then on every explore_every-th cycle (0: never)."""
    if np.asarray(error_set).size == 0:
        return True
    return explore_every > 0 and cycle is not None and cycle % explore_every == 0


def step_2(archive, last_test_set, sigma_value, rng=None, sampler=None, cycle=None):
    """README "Algorithm", item 5. Step 2: propose the next test set from the archive, for the
    cycle number `cycle`. While E is empty there is nothing to exploit (coverage mode): every
    slot is an exploration slot (5.5) and 5.1-5.4 are skipped. From the first failure on, every
    explore_every-th cycle is a coverage cycle as well (when `cycle` is given); in the other
    cycles popsize - explore_slots slots exploit E (5.1-5.4) and explore_slots slots explore."""
    if rng is None:
        rng = globals()["rng"]
    if sampler is None:
        sampler = HaltonSampler(rng)

    error_set = archive_error_set(archive)
    passed_set = archive_passed_set(archive)
    tested_set = archive_tested_set(archive)

    if is_coverage_cycle(error_set, cycle):
        n_explore = popsize
        chosen = np.empty((0, 2), dtype=int)
    else:
        n_explore = min(explore_slots, popsize)
        chosen = _exploit(
            error_set, passed_set, tested_set, last_test_set, sigma_value, popsize - n_explore,
            rng, sampler,
        )

    # 5.5 Exploration slots: farthest-point picks from a fresh pool, so the domain is covered
    # evenly while E is empty, and failure regions far from E can still be found afterwards.
    # The pool is topped up (up to max_attempts rounds of explore_pool draws) until it holds
    # enough tuples that are neither duplicates nor already tested or chosen.
    if n_explore > 0:
        taken = np.concatenate((tested_set, chosen), axis=0)
        pool = np.empty((0, 2), dtype=int)
        for _ in range(max_attempts):
            pool = np.unique(
                np.concatenate((pool, sampler.draw(explore_pool)), axis=0),
                axis=0,
            )
            if taken.size:
                already_taken = (pool[:, None, :] == taken[None, :, :]).all(axis=2).any(axis=1)
                pool = pool[~already_taken]
            if pool.shape[0] >= n_explore:
                break
        chosen = np.concatenate((chosen, select_explorers(pool, taken, n_explore)), axis=0)
    return chosen


def run_simulation(oracle, cycles, seed=None, initial_test_set=None):
    """Run `cycles` EA cycles (Step 1 included), one HiL test run each, numbered as in README
    "Algorithm": Step 1 (or the given initial test set), then repeatedly run the test set on
    the oracle, record the verdicts, adapt sigma and propose the next set (Step 2). Ends after
    `cycles` cycles ("cycles done"), or earlier when Step 2 finds no untested candidate
    ("no more candidates"). A given `seed` drives a generator of this run only: the module's
    `rng` and `seed` stay untouched. Without one, the run draws from the module's `rng`."""
    errors, _ = check_valid_parameters()
    if errors:
        raise ValueError(" ".join(errors))
    # Logged seed: the one behind the generator this run draws from, given here or set earlier
    # with seed_rng() (None if the module's rng was never seeded).
    if seed is not None:
        local_rng, run_seed = np.random.default_rng(seed), seed
    else:
        local_rng, run_seed = globals()["rng"], globals()["seed"]

    archive = new_archive()
    if error_hit_set.size:
        archive = merge_manual_error_hit_set(archive, error_hit_set)

    # 1. Step 1
    # One Halton sequence per run, continued across cycles (its random shift is drawn first).
    sampler = HaltonSampler(local_rng)
    test_set = (
        initial_test_set if initial_test_set is not None else step_1(rng=local_rng, sampler=sampler)
    )

    sigma_value = sigma
    cycles_without_failure = 0
    log_rows = []
    termination_reason = "cycles done"

    for cycle in range(1, cycles + 1):
        # Logged fitness: each tuple scored against the archive *before* this cycle, i.e. the
        # fitness it had when it was proposed (scored after recording, every tuple would be
        # its own nearest tested tuple: V3 = 1, and V1 = 0 for every failure).
        v1, v2, v3 = fitness_components(
            test_set, archive_error_set(archive), archive_tested_set(archive)
        )
        f = v1 + gamma * v2 + beta * v3

        # 2. Run, 3. Record
        verdicts = np.array([oracle(tuple(row)) for row in test_set])
        archive = record_results(archive, test_set, verdicts, cycle)
        for i, row in enumerate(test_set):
            log_rows.append(
                {
                    "cycle": cycle,
                    "t_start": row[t_start_column],
                    "t_end": row[t_end_column],
                    "V1": v1[i],
                    "V2": v2[i],
                    "V3": v3[i],
                    "f": f[i],
                    "sigma": sigma_value,
                    "seed": run_seed,
                    "verdict": verdicts[i],
                }
            )

        # 4. Adapt sigma
        sigma_value, cycles_without_failure = update_sigma(
            sigma_value, bool(verdicts.sum() > 0), cycles_without_failure
        )

        # 5. Step 2, unless this was the last cycle asked for
        if cycle == cycles:
            break
        test_set = step_2(
            archive, test_set, sigma_value, rng=local_rng, sampler=sampler, cycle=cycle + 1
        )
        if test_set.shape[0] == 0:
            termination_reason = "no more candidates"
            break

    log_df = pd.DataFrame(log_rows)

    return archive, log_df, termination_reason
