import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Evolutionary_Algorithm as ea


@pytest.fixture(autouse=True)
def reset_params():
    """Every test runs against the module defaults, regardless of what earlier tests changed."""
    saved = {
        name: getattr(ea, name)
        for name in (
            "T_Start", "t_0", "w", "gamma", "r", "beta", "s", "sigma", "sigma_min",
            "sigma_max", "patience", "immigrant_rate", "d_min", "popsize", "lambda_new",
            "resolution_ms", "max_cycles", "offset", "explore_slots",
        )
    }
    yield
    for name, value in saved.items():
        setattr(ea, name, value)


# --- V1 ---

def test_v1_zero_at_known_tuples_and_cross_combinations():
    error_set = np.array([[-4025, -300], [-3213, -1232]])
    samples = np.array([[-4025, -300], [-3213, -1232], [-4025, -1232], [-3213, -300]])
    v1 = ea.V_1(samples, error_set)
    assert v1 == pytest.approx(0, abs=1e-9)


def test_v1_zero_when_error_set_empty():
    samples = np.array([[-2000, -500], [-100, -50]])
    v1 = ea.V_1(samples, np.array([]))
    assert np.allclose(v1, 0)


# --- V2 ---

def test_v2_zero_inside_t():
    samples = np.array([[-1000, -500]])
    v2 = ea.V_2(samples)
    assert v2[0] == pytest.approx(0)


def test_v2_distance_outside_t_with_nonzero_t0():
    ea.t_0 = 100
    samples = np.array([[-1000, 50], [-1000, 150]])
    v2 = ea.V_2(samples)
    abs_T = abs(ea.t_0 - ea.T_Start)
    assert v2[0] == pytest.approx(0)
    assert v2[1] == pytest.approx(50 / abs_T)


# --- V3 ---

def test_v3_ring_minimum_at_radius_r():
    e = np.array([-2000, -500])
    ea.beta = 0.05
    ea.r = 0.01
    abs_T = abs(ea.t_0 - ea.T_Start)

    f_at_e = ea.fitness_function(e.reshape(1, 2), e.reshape(1, 2), e.reshape(1, 2))
    assert f_at_e[0] == pytest.approx(ea.beta, abs=1e-6)

    # Walk outward along t_End from e: f should dip below f(e) and reach its minimum near r*|T|,
    # then rise again beyond it.
    offsets = np.arange(0, int(ea.r * abs_T) * 3, max(1, int(ea.r * abs_T) // 10))
    probe = np.stack([np.full_like(offsets, e[0]), e[1] + offsets], axis=1)
    f = ea.fitness_function(probe, e.reshape(1, 2), e.reshape(1, 2))
    min_idx = np.argmin(f)
    assert f[min_idx] < f_at_e[0]
    assert f[-1] > f[min_idx]


# --- Selection ---

def test_rank_probabilities_sum_to_one_and_handle_zero_scores():
    scores = np.array([0.0, 0.0, 0.3, 1.2, 0.0])
    p = ea.rank_probabilities(scores)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)


def test_rank_probabilities_best_gets_highest_p():
    scores = np.array([5.0, 0.1, 3.0, 1.0])
    p = ea.rank_probabilities(scores)
    assert np.argmax(p) == np.argmin(scores)


# --- Candidates ---

def test_generate_candidates_avoid_duplicates_and_tested_and_respect_d_min():
    rng = np.random.default_rng(1)
    parents = np.array([[-2000, -500], [-1500, -300]])
    tested = np.array([[-2000, -500], [-1500, -300]])
    candidates = ea.generate_candidates(parents, tested, sigma_value=20, n=10, rng=rng)

    assert candidates.shape[0] > 0
    assert len(np.unique(candidates, axis=0)) == candidates.shape[0]
    for row in candidates:
        assert not (tested == row).all(axis=1).any()
    assert np.all(candidates[:, ea.t_end_column] - candidates[:, ea.t_start_column] >= ea.d_min)


def test_generate_candidates_sigma_zero_does_not_crash_or_hang():
    rng = np.random.default_rng(2)
    parents = np.array([[-2000, -500]])
    tested = np.array([[-2000, -500]])
    candidates = ea.generate_candidates(parents, tested, sigma_value=0, n=5, rng=rng)
    assert isinstance(candidates, np.ndarray)
    assert candidates.shape[1] == 2


# --- Greedy test-set selection ---

def test_select_test_set_returns_distinct_untested_tuples():
    rng = np.random.default_rng(3)
    candidates = ea.initialize_samples_array(50, rng=rng)
    tested = np.array([[-2000, -500]])
    chosen = ea.select_test_set(candidates, np.array([]), tested, n=4)
    assert chosen.shape[0] == 4
    assert len(np.unique(chosen, axis=0)) == 4
    for row in chosen:
        assert not (tested == row).all(axis=1).any()


def test_select_test_set_spaces_candidates_apart():
    # Two near-identical candidates plus a well-spread alternative: the greedy pick should take
    # at most one of the near-identical pair once the first is chosen (V3 penalises the second).
    ea.beta = 0.05
    ea.r = 0.01
    candidates = np.array([[-2000, -500], [-2001, -499], [-500, -4000]])
    chosen = ea.select_test_set(candidates, np.array([]), np.array([]), n=2)
    picked = {tuple(row) for row in chosen}
    close_pair_both_picked = (-2000, -500) in picked and (-2001, -499) in picked
    assert not close_pair_both_picked


# --- Exploration slots ---

def test_select_explorers_picks_the_farthest_tuple_and_spreads_out():
    tested = np.array([[-4900, -4000]])
    pool = np.array([[-100, -50], [-110, -60], [-2500, -1000]])
    first = ea.select_explorers(pool, tested, n=1)
    assert tuple(first[0]) == (-100, -50)
    # The second pick skips the first pick's neighbour: it is too close to the first pick.
    both = ea.select_explorers(pool, tested, n=2)
    assert {tuple(row) for row in both} == {(-100, -50), (-2500, -1000)}


@pytest.mark.parametrize("slots", [0, 1, 4])
def test_step_2_returns_popsize_distinct_untested_tuples_with_explore_slots(slots):
    ea.explore_slots = slots
    archive = ea.record_results(ea.new_archive(), ea.step_1(), np.array([1, 0, 0, 0]), cycle=1)
    proposed = ea.step_2(archive, ea.step_1(), ea.sigma, rng=np.random.default_rng(5))
    tested = ea.archive_tested_set(archive)
    assert proposed.shape[0] == ea.popsize
    assert len(np.unique(proposed, axis=0)) == ea.popsize
    for row in proposed:
        assert not (tested == row).all(axis=1).any()


# --- Reproducibility ---

def test_same_seed_gives_identical_simulation_log():
    def oracle(sample):
        t_start, t_end = sample
        return 1 if abs(t_start + 3000) < 200 and abs(t_end + 300) < 100 else 0

    _, log_a, reason_a = ea.run_simulation(
        oracle, max_cycles=5, seed=123, initial_test_set=ea.step_1(), log_dir=None, archive_path=None
    )
    _, log_b, reason_b = ea.run_simulation(
        oracle, max_cycles=5, seed=123, initial_test_set=ea.step_1(), log_dir=None, archive_path=None
    )
    assert reason_a == reason_b
    assert log_a.equals(log_b)


def test_simulation_never_tests_a_tuple_twice():
    def oracle(sample):
        t_start, t_end = sample
        return 1 if abs(t_start + 3000) < 200 and abs(t_end + 300) < 100 else 0

    archive, _, _ = ea.run_simulation(
        oracle, max_cycles=8, seed=7, initial_test_set=ea.step_1(), log_dir=None, archive_path=None
    )
    assert archive.duplicated(subset=["t_start", "t_End"]).sum() == 0
