import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# For every matrix containing samples in the way (t_start, t_End) the columns are numerated:
t_start_column = 0
t_end_column = 1

# T_Start: Instant when pre-crash recording time starts.
T_Start = -5000
# t_0: Instant when pre-crash recording time ends.
# As t_0 is the time when the recording algorithm starts, it is normally set to zero.
t_0 = 0
# T: Pre-crash recording time established in the regulation.
T = (T_Start, t_0)
# Possible offset of the range (T_Start, t_0) for the creation of samples in step_1
offset = 10

# popsize: size of a test set (EA's output, run on the HiL as one batch)
popsize = 4
# lambda_new: number of candidates generated each cycle, before the greedy test-set selection
lambda_new = 5 * popsize
# max_attempts: retry rounds for candidate generation, so a cycle that only produces duplicates
# does not loop forever
max_attempts = 20

# --- Fitness weights (all terms are normalised by |T|, so they are unitless) ---
# w: weight of t_End vs t_start in V1 and in the whole-tuple distance d. w > 0.5 means
# "t_End matters more than t_start".
w = 0.75
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
sigma_max = abs(T_Start - t_0) / 10
# patience: cycles without a new failure before sigma grows (explore further).
patience = 2
# immigrant_rate: fraction of each candidate batch drawn fresh from initialize_samples_array()
# instead of from mutation, to keep exploring (Pesticide Paradox).
immigrant_rate = 0.2
# explore_slots: test-set slots reserved for pure exploration (see select_explorers). Without
# them, once E is non-empty V1 pulls every slot toward the known errors, and failure regions
# far from them are practically never found.
explore_slots = 1
# d_min: minimum event duration (t_End - t_start), so a mutated tuple is never zero-length.
d_min = 1

# --- Termination ---
# resolution_ms: boundary resolution; a cycle stops proposing boundary candidates once every
# error/pass pair is closer than this.
resolution_ms = 1
# max_cycles: HiL cycle limit for run_simulation.
max_cycles = 4

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


# error_hit_set: (Fehlertreff in Excel sheet) tuples manually known to have found an error,
# merged into a fresh archive as fail rows with cycle = 0 (see merge_manual_error_hit_set).
error_hit_set = np.array([])
# first_test_samples / test_samples: kept for interactive/manual use (CLI, notebook); the
# WebApp keeps its own state in st.session_state instead (see WebApp/layout.py).
first_test_samples = np.array([])
test_samples = np.array([])


def _abs_T():
    return abs(t_0 - T_Start)


def _repair(samples):
    """Clip into the valid domain [T_Start-offset, t_0+offset], swap t_start/t_End if
    inverted, then enforce the minimum event duration d_min. The domain clip keeps mutation
    from drifting candidates away from the test case indefinitely during pure novelty search
    (V1 = 0, nothing else pulls t_start back); t_start may still lie before T_Start, as long
    as it stays within `offset` of it."""
    samples = np.asarray(samples, dtype=int).reshape(-1, 2).copy()
    if samples.shape[0] == 0:
        return samples
    lo, hi = T_Start - offset, t_0 + offset
    samples = np.clip(samples, lo, hi)
    swap = samples[:, t_start_column] > samples[:, t_end_column]
    samples[swap] = samples[swap][:, ::-1]
    too_short = (samples[:, t_end_column] - samples[:, t_start_column]) < d_min
    samples[too_short, t_end_column] = samples[too_short, t_start_column] + d_min
    return samples


def initialize_samples_array(n=None, rng=None):
    # Generate random test samples creation (Step 1): n tuples drawn uniformly in the
    # pre-crash recording time T extended by `offset` on both sides. t_start may lie before
    # T_Start (only t_End is penalised, by V2); duration is repaired to at least d_min.
    if n is None:
        n = popsize
    if rng is None:
        rng = globals()["rng"]
    lo, hi = T_Start - offset, t_0 + offset
    samples = rng.uniform(lo, hi, size=(n, 2))
    samples = np.rint(samples).astype(int)
    return _repair(samples)


def _weighted_distance_ms(a, b):
    """Pairwise weighted distance in ms between every row of a and every row of b.
    Shape (len(a), len(b)). weight w favours t_End over t_start."""
    a = np.asarray(a).reshape(-1, 2)
    b = np.asarray(b).reshape(-1, 2)
    diff = np.abs(a[:, None, :] - b[None, :, :])
    return (1 - w) * diff[:, :, t_start_column] + w * diff[:, :, t_end_column]


def distance_to_set(samples, ref):
    """Normalised whole-tuple distance d(x, y) from each sample to the nearest row in ref."""
    samples = np.asarray(samples).reshape(-1, 2)
    ref = np.asarray(ref).reshape(-1, 2)
    if ref.size == 0:
        return np.full(samples.shape[0], np.inf)
    return _weighted_distance_ms(samples, ref).min(axis=1) / _abs_T()


def V_1(samples_array, error_set):
    # t_start and t_End are independent coordinates: the nearest E_i[t_start] and the nearest
    # E_j[t_End] are taken separately, so they may come from different tuples of error_set
    # (see README "Fitness Orientation"). 0 if error_set is empty.
    samples_array = np.asarray(samples_array).reshape(-1, 2)
    error_set = np.asarray(error_set).reshape(-1, 2)
    if error_set.size == 0:
        return np.zeros(samples_array.shape[0])

    diff = np.abs(samples_array[:, None, :] - error_set[None, :, :])
    nearest = diff.min(axis=1)  # shape (n, 2): per-coordinate minimum distance
    weighted = (1 - w) * nearest[:, t_start_column] + w * nearest[:, t_end_column]
    return weighted / _abs_T()


def V_2(samples_array):
    # Soft penalty for t_End lying outside T = [T_Start, t_0], normalised by |T|.
    samples_array = np.asarray(samples_array).reshape(-1, 2)
    t_end = samples_array[:, t_end_column]
    distance = np.maximum(T_Start - t_end, t_end - t_0)
    return distance.clip(min=0) / _abs_T()


def V_3(samples_array, tested_set):
    # Novelty term (Pesticide Paradox): repels samples from everything already tested (A).
    # Whole-tuple distance, so it never forbids reusing a single known coordinate.
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
    # Linear-ranking roulette: rank 0 is the worst (highest f), rank n-1 the best (lowest f).
    # p = (2-s)/n + 2*rank*(s-1)/(n*(n-1)), s in [1, 2]. Scale-free, safe for f = 0.
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


def select_offspring(bins, samples, n, rng=None):
    if rng is None:
        rng = globals()["rng"]
    x = rng.random(n)
    bins_cumulative = np.cumsum(bins)
    selection = np.digitize(x, bins_cumulative)
    selection = np.clip(selection, 0, samples.shape[0] - 1)
    return samples[selection, :]


def update_sigma(sigma_value, new_failures, cycles_without_failure):
    # Adaptive step size driven by HiL feedback: shrink to exploit a fresh failure, grow after
    # `patience` quiet cycles to explore again. Clipped to [sigma_min, sigma_max].
    if new_failures:
        sigma_value = sigma_value * 0.7
        cycles_without_failure = 0
    else:
        cycles_without_failure += 1
        if cycles_without_failure >= patience:
            sigma_value = sigma_value * 1.5
            cycles_without_failure = 0
    sigma_value = float(np.clip(sigma_value, sigma_min, sigma_max))
    return sigma_value, cycles_without_failure


def generate_candidates(parents, tested, sigma_value, n=None, rng=None):
    # Normal mutation of selected parents, plus a share of random immigrants, repaired and
    # de-duplicated against each other and against everything already tested. Retries up to
    # max_attempts rounds so a batch that only produces duplicates does not stall the cycle.
    if n is None:
        n = lambda_new
    if rng is None:
        rng = globals()["rng"]
    tested = np.asarray(tested).reshape(-1, 2)
    parents = np.asarray(parents).reshape(-1, 2)

    collected = np.empty((0, 2), dtype=int)
    for _ in range(max_attempts):
        n_immigrants = int(round(n * immigrant_rate))
        n_mutants = n - n_immigrants

        mutants = np.empty((0, 2), dtype=int)
        if n_mutants > 0 and parents.shape[0] > 0:
            parent_idx = rng.integers(0, parents.shape[0], size=n_mutants)
            mutation = np.rint(rng.normal(0, sigma_value, size=(n_mutants, 2))).astype(int)
            mutants = parents[parent_idx] + mutation

        immigrants = (
            initialize_samples_array(n_immigrants, rng=rng)
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
    # Greedy set selection (comma selection: tested tuples never return). Repeat n times: score
    # every remaining candidate against A u chosen, take the lowest, add it. V3 then also
    # spaces the chosen tuples apart, so the set spreads over the error clusters.
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
        best_idx = int(np.argmin(scores))
        chosen = np.concatenate((chosen, remaining[best_idx : best_idx + 1]), axis=0)
        remaining = np.delete(remaining, best_idx, axis=0)

    if chosen.shape[0] < n:
        warnings.warn(f"Only {chosen.shape[0]} candidates available; requested {n}.")
    return chosen


def select_explorers(pool, tested_set, n=None):
    # Pure exploration, no V1: greedy farthest-point selection. Repeat n times: take the pool
    # tuple farthest (by d) from A u chosen, with V2 still keeping t_End inside T. Each pick
    # lands in the largest untested gap, so over the cycles the explorers cover T evenly.
    if n is None:
        n = explore_slots
    pool = np.asarray(pool).reshape(-1, 2)
    tested_set = np.asarray(tested_set).reshape(-1, 2)

    chosen = np.empty((0, 2), dtype=int)
    remaining = pool.copy()
    for _ in range(min(n, remaining.shape[0])):
        reference = np.concatenate((tested_set, chosen), axis=0)
        scores = gamma * V_2(remaining) - distance_to_set(remaining, reference)
        best_idx = int(np.argmin(scores))
        chosen = np.concatenate((chosen, remaining[best_idx : best_idx + 1]), axis=0)
        remaining = np.delete(remaining, best_idx, axis=0)
    return chosen


def boundary_candidates(error_set, passed_set, resolution=None):
    # For each error e, take the nearest passed tuple p. If d(e, p) is wider than `resolution`,
    # propose the rounded midpoint as a candidate: it competes like any other candidate, and
    # passed tuples repel through V3, so the search fills in the boundary region.
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


def boundary_resolved(error_set, passed_set, resolution=None):
    error_set = np.asarray(error_set).reshape(-1, 2)
    passed_set = np.asarray(passed_set).reshape(-1, 2)
    if error_set.size == 0 or passed_set.size == 0:
        return False
    return boundary_candidates(error_set, passed_set, resolution).shape[0] == 0


# --- Archive: a DataFrame of every tuple run on the HiL so far, with its verdict and cycle. ---
ARCHIVE_COLUMNS = ["t_start", "t_End", "verdict", "cycle"]


def new_archive():
    return pd.DataFrame(columns=ARCHIVE_COLUMNS)


def record_results(archive, test_set, verdicts, cycle):
    test_set = np.asarray(test_set).reshape(-1, 2)
    verdicts = np.asarray(verdicts).astype(int)
    rows = pd.DataFrame(
        {
            "t_start": test_set[:, t_start_column],
            "t_End": test_set[:, t_end_column],
            "verdict": verdicts,
            "cycle": cycle,
        }
    )
    return pd.concat([archive, rows], ignore_index=True)


def archive_error_set(archive):
    if archive.empty:
        return np.empty((0, 2), dtype=int)
    fails = archive[archive["verdict"] == 1]
    return fails[["t_start", "t_End"]].to_numpy(dtype=int)


def archive_passed_set(archive):
    if archive.empty:
        return np.empty((0, 2), dtype=int)
    passed = archive[archive["verdict"] == 0]
    return passed[["t_start", "t_End"]].to_numpy(dtype=int)


def archive_tested_set(archive):
    if archive.empty:
        return np.empty((0, 2), dtype=int)
    return archive[["t_start", "t_End"]].to_numpy(dtype=int)


def merge_manual_error_hit_set(archive, manual_error_hit_set):
    # A manually entered error_hit_set (E) is merged in as fail rows with cycle = 0.
    manual = np.asarray(manual_error_hit_set).reshape(-1, 2)
    if manual.size == 0:
        return archive
    rows = pd.DataFrame(
        {
            "t_start": manual[:, t_start_column],
            "t_End": manual[:, t_end_column],
            "verdict": 1,
            "cycle": 0,
        }
    )
    merged = pd.concat([archive, rows], ignore_index=True)
    return merged.drop_duplicates(subset=["t_start", "t_End"], keep="first").reset_index(drop=True)


# 1. Random test samples creation:
#  step_1() intentionally returns a fixed example population, so the CLI demo and the tests
#  are reproducible without relying on initialize_samples_array()'s randomness.
def step_1():
    samples = np.array([[-2420, -259], [-4338, -594], [-3616, -728], [-1311, -104]])
    return samples


# 2. Test execution and following test sample creation process (propose the next test set).
def step_2(archive, last_test_set, sigma_value, rng=None):
    if rng is None:
        rng = globals()["rng"]

    error_set = archive_error_set(archive)
    passed_set = archive_passed_set(archive)
    tested_set = archive_tested_set(archive)

    # 2.1 Parent pool: last test set u E, so every error cluster stays alive even when it
    # wasn't part of the last test set.
    parent_pool = np.unique(
        np.concatenate((np.asarray(last_test_set).reshape(-1, 2), error_set), axis=0), axis=0
    )
    if parent_pool.size == 0:
        parent_pool = initialize_samples_array(popsize, rng=rng)

    # 2.2 Offspring: linear-ranking roulette selects lambda_new parents.
    scores = fitness_function(parent_pool, error_set, tested_set)
    p = rank_probabilities(scores)
    selected_parents = select_offspring(p, parent_pool, lambda_new, rng=rng)

    # 2.3 Mutation (+ immigrants) and boundary candidates (A3).
    candidates = generate_candidates(selected_parents, tested_set, sigma_value, rng=rng)
    extra = boundary_candidates(error_set, passed_set)
    if extra.size:
        if tested_set.size:
            already_tested = (extra[:, None, :] == tested_set[None, :, :]).all(axis=2).any(axis=1)
            extra = extra[~already_tested]
        candidates = np.unique(np.concatenate((candidates, extra), axis=0), axis=0)

    # 2.4-2.5 Greedy test-set selection (A7): the next test set, picked from candidates only,
    # with explore_slots of it left for 2.6.
    n_explore = min(explore_slots, popsize)
    chosen = select_test_set(candidates, error_set, tested_set, n=popsize - n_explore)

    # 2.6 Exploration slots: farthest-point picks from a fresh random pool, so failure regions
    # far from E can still be found while the rest of the set exploits E.
    if n_explore > 0:
        pool = initialize_samples_array(lambda_new, rng=rng)
        taken = np.concatenate((tested_set, chosen), axis=0)
        if taken.size:
            already_taken = (pool[:, None, :] == taken[None, :, :]).all(axis=2).any(axis=1)
            pool = pool[~already_taken]
        pool = np.unique(pool, axis=0)
        chosen = np.concatenate((chosen, select_explorers(pool, taken, n_explore)), axis=0)
    return chosen


def run_simulation(
    oracle,
    max_cycles=None,
    seed=None,
    initial_test_set=None,
    log_dir="runs",
    archive_path="archive.csv",
):
    # One EA cycle = one HiL test run: Step 1 (or the given initial test set), then repeatedly
    # run the test set on the oracle, record verdicts, adapt sigma, and propose the next set,
    # until the failure boundary is resolved or max_cycles is reached.
    if max_cycles is None:
        max_cycles = globals()["max_cycles"]
    local_rng = seed_rng(seed) if seed is not None else globals()["rng"]

    archive = new_archive()
    if error_hit_set.size:
        archive = merge_manual_error_hit_set(archive, error_hit_set)

    test_set = (
        initial_test_set
        if initial_test_set is not None
        else initialize_samples_array(popsize, rng=local_rng)
    )

    sigma_value = sigma
    cycles_without_failure = 0
    log_rows = []
    termination_reason = "max_cycles"

    for cycle in range(1, max_cycles + 1):
        verdicts = np.array([oracle(tuple(row)) for row in test_set])
        archive = record_results(archive, test_set, verdicts, cycle)

        error_set = archive_error_set(archive)
        passed_set = archive_passed_set(archive)
        tested_set = archive_tested_set(archive)
        v1, v2, v3 = fitness_components(test_set, error_set, tested_set)
        f = v1 + gamma * v2 + beta * v3
        for i, row in enumerate(test_set):
            log_rows.append(
                {
                    "cycle": cycle,
                    "t_start": row[t_start_column],
                    "t_End": row[t_end_column],
                    "V1": v1[i],
                    "V2": v2[i],
                    "V3": v3[i],
                    "f": f[i],
                    "sigma": sigma_value,
                    "seed": seed,
                    "verdict": verdicts[i],
                }
            )

        sigma_value, cycles_without_failure = update_sigma(
            sigma_value, bool(verdicts.sum() > 0), cycles_without_failure
        )

        if boundary_resolved(error_set, passed_set):
            termination_reason = "boundary resolved"
            break

        test_set = step_2(archive, test_set, sigma_value, rng=local_rng)
        if test_set.shape[0] == 0:
            termination_reason = "no more candidates"
            break

    log_df = pd.DataFrame(log_rows)
    if log_dir is not None:
        Path(log_dir).mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_df.to_csv(Path(log_dir) / f"{timestamp}_log.csv", index=False)
    if archive_path is not None:
        archive.to_csv(archive_path, index=False)

    return archive, log_df, termination_reason


if __name__ == "__main__":
    import simulation

    simulation.main()
