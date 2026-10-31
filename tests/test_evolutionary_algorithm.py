import numpy as np
import pytest

import Evolutionary_Algorithm as ea
from WebApp import simulation


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

    # Walk outward along t_end from e: f should dip below f(e) and reach its minimum near r*|T|,
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
    candidates = ea.generate_candidates(parents, tested, sigma_value=20, n_immigrants=8, rng=rng)

    assert candidates.shape[0] > 0
    assert len(np.unique(candidates, axis=0)) == candidates.shape[0]
    for row in candidates:
        assert not (tested == row).all(axis=1).any()
    assert np.all(candidates[:, ea.t_end_column] - candidates[:, ea.t_start_column] >= ea.d_min)


def test_generate_candidates_sigma_zero_does_not_crash_or_hang():
    rng = np.random.default_rng(2)
    parents = np.array([[-2000, -500]])
    tested = np.array([[-2000, -500]])
    candidates = ea.generate_candidates(parents, tested, sigma_value=0, n_immigrants=4, rng=rng)
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
    """Two near-identical candidates plus a well-spread alternative: the greedy pick should take
    at most one of the near-identical pair once the first is chosen (V3 penalises the second)."""
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
    first = ea.step_1(rng=np.random.default_rng(4))
    archive = ea.record_results(ea.new_archive(), first, np.array([1, 0, 0, 0]), cycle=1)
    proposed = ea.step_2(archive, first, ea.sigma, rng=np.random.default_rng(5))
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
        oracle, cycles=5, seed=123
    )
    _, log_b, reason_b = ea.run_simulation(
        oracle, cycles=5, seed=123
    )
    assert reason_a == reason_b
    assert log_a.equals(log_b)


def test_simulation_never_tests_a_tuple_twice():
    def oracle(sample):
        t_start, t_end = sample
        return 1 if abs(t_start + 3000) < 200 and abs(t_end + 300) < 100 else 0

    archive, _, _ = ea.run_simulation(
        oracle, cycles=8, seed=7
    )
    assert archive.duplicated(subset=["t_start", "t_end"]).sum() == 0


# --- Step 1 ---

def test_step_1_is_random_popsize_tuples_reproducible_by_rng():
    ea.popsize = 6
    a = ea.step_1(rng=np.random.default_rng(11))
    b = ea.step_1(rng=np.random.default_rng(11))
    assert a.shape == (6, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ea.step_1(rng=np.random.default_rng(12)))


# --- Parent selection ---

def test_select_parents_follows_the_probabilities():
    pool = np.array([[-3000, -100], [-2000, -100], [-1000, -100]])
    parents = ea.select_parents(np.array([0.0, 1.0, 0.0]), pool, 5, rng=np.random.default_rng(0))
    assert parents.shape == (5, 2)
    assert (parents == pool[1]).all()


def test_generate_candidates_mutates_each_parent_once_per_round():
    """sigma = 0 and no immigrants: a single round yields exactly the (repaired) parents."""
    rng = np.random.default_rng(8)
    parents = np.array([[-3000, -100], [-2000, -100]])
    candidates = ea.generate_candidates(parents, np.array([]), sigma_value=0, n_immigrants=0, rng=rng)
    assert {tuple(row) for row in candidates} == {tuple(row) for row in parents}


# --- Repair ---

def test_repair_stays_in_domain_when_d_min_stretch_hits_the_upper_bound():
    lo, hi = ea.T_Start - ea.offset, ea.t_0 + ea.offset
    repaired = ea._repair([[hi, hi], [hi + 50, hi + 80], [lo, lo]])
    assert repaired.min() >= lo and repaired.max() <= hi
    assert np.all(repaired[:, ea.t_end_column] - repaired[:, ea.t_start_column] >= ea.d_min)


# --- Archive ---

def test_archive_columns_are_typed():
    archive = ea.merge_manual_error_hit_set(ea.new_archive(), np.array([[-4025, -300]]))
    archive = ea.record_results(archive, np.array([[-2000, -500]]), np.array([0]), cycle=1)
    assert list(archive.columns) == ["t_start", "t_end", "verdict", "cycle", "source"]
    numeric = archive[["t_start", "t_end", "verdict", "cycle"]]
    assert all(np.issubdtype(dtype, np.integer) for dtype in numeric.dtypes)
    assert archive["source"].tolist() == ["manual", "hil"]


def test_manual_failures_are_merged_at_the_given_cycle_and_never_removed():
    archive = ea.record_results(
        ea.new_archive(), np.array([[-2000, -500], [-1000, -100]]), np.array([1, 0]), cycle=1
    )
    archive = ea.record_results(archive, np.array([[-3000, -700]]), np.array([0]), cycle=3)
    merged = ea.merge_manual_error_hit_set(
        archive,
        np.array([[-2000, -500], [-1000, -100], [-3000, -700], [-4025, -300], [-4025, -300]]),
        cycle=3,
    )
    rows = {(r.t_start, r.t_end): (r.verdict, r.cycle, r.source) for r in merged.itertuples()}
    assert rows == {
        (-2000, -500): (1, 1, "hil"),  # already in E: unchanged
        (-1000, -100): (1, 3, "override"),  # earlier pass: marked failed, a new failure of cycle 3
        (-3000, -700): (1, 3, "hil"),  # pass of the current test set: like checking its box
        (-4025, -300): (1, 3, "manual"),  # new: added once, at the given cycle
    }
    assert ea.archive_error_set(merged).shape[0] == 4


# --- error_hit_set input ---

def test_parse_error_hit_set_accepts_valid_lines_and_skips_blank_ones():
    parsed = ea.parse_error_hit_set("-4025, -300\n\n-3213, -1232\n-4025, -300\n")
    assert parsed.tolist() == [[-4025, -300], [-3213, -1232]]
    assert ea.parse_error_hit_set("  \n").shape == (0, 2)


@pytest.mark.parametrize(
    "text",
    ["-4025, -300, -10", "-4025", "-4025, abc", "-300, -4025", "-100, -100", "-9000, -300", "-100, 500"],
)
def test_parse_error_hit_set_rejects_malformed_lines(text):
    with pytest.raises(ValueError, match="line 1"):
        ea.parse_error_hit_set(text)


# --- Simulation ---

def test_simulated_oracle_regions():
    assert simulation.oracle((-4025, -300)) == 1  # region_1, first example error_hit_set tuple
    assert simulation.oracle((-4800, -2000)) == 1  # region_2
    assert simulation.oracle((-3213, -1232)) == 0  # second example tuple: a format example only


def test_log_scores_each_tuple_against_the_archive_before_its_cycle():
    def oracle(sample):
        """A large failure region (t_end late in T), so E fills up within a few cycles."""
        return 1 if sample[1] > -2500 else 0

    archive, log, _ = ea.run_simulation(oracle, cycles=6, seed=3)
    assert (log["V1"] > 0).any() and (log["V3"] < 1).any()
    for cycle, rows in log.groupby("cycle"):
        before = archive[archive["cycle"] < cycle]
        v1, v2, v3 = ea.fitness_components(
            rows[["t_start", "t_end"]].to_numpy(),
            ea.archive_error_set(before),
            ea.archive_tested_set(before),
        )
        assert np.allclose(rows["V1"], v1) and np.allclose(rows["V3"], v3)


def test_log_records_the_seed_set_earlier_with_seed_rng():
    ea.seed_rng(9)
    _, log, _ = ea.run_simulation(simulation.oracle, cycles=2)
    assert (log["seed"] == 9).all()


# --- Adaptive sigma ---

def test_sigma_upper_bound_follows_the_domain():
    """Many quiet cycles push sigma up to its bound, sigma_max_ratio * |T|, for the current T."""
    for T_Start in (-5000, -20000):
        ea.T_Start = T_Start
        sigma_value, quiet = ea.sigma, 0
        for _ in range(50):
            sigma_value, quiet = ea.update_sigma(sigma_value, False, quiet)
        assert sigma_value == pytest.approx(ea.sigma_max_ratio * abs(ea.t_0 - T_Start))


# --- Termination ---

def test_run_goes_exactly_the_given_cycles():
    archive, _, reason = ea.run_simulation(lambda sample: 0, cycles=15, seed=1)
    assert reason == "cycles done"
    assert archive["cycle"].max() == 15


@pytest.mark.filterwarnings("ignore:Only .* candidates available")  # expected near the end
def test_run_ends_with_no_more_candidates_on_a_tiny_domain():
    # [-6, 0] holds 21 possible tuples: the run tests each once, then Step 2 has nothing left.
    ea.T_Start, ea.t_0, ea.offset = -6, 0, 0
    ea.sigma_max_ratio = 1  # sigma up to |T| = 6 ms, so that sigma_min (1 ms) fits below it
    for seed in range(3):
        archive, _, reason = ea.run_simulation(lambda sample: 0, cycles=50, seed=seed)
        assert reason == "no more candidates"
        assert archive["cycle"].max() < 50
        assert len(archive) == 21
        assert archive.duplicated(subset=["t_start", "t_end"]).sum() == 0


# --- Parameter checks ---

def test_check_valid_parameters_and_run_refused_on_errors():
    assert ea.check_valid_parameters() == ([], [])  # the defaults are consistent

    ea.popsize, ea.lambda_new, ea.explore_slots = 30, 20, 1
    errors, _ = ea.check_valid_parameters()
    assert len(errors) == 1 and "lambda_new" in errors[0]
    with pytest.raises(ValueError, match="lambda_new"):
        ea.run_simulation(simulation.oracle, cycles=1, seed=1)
    ea.lambda_new = 29  # = popsize - explore_slots: allowed
    assert ea.check_valid_parameters()[0] == []

    ea.T_Start, ea.t_0 = 0, 0
    assert any("T_Start" in message for message in ea.check_valid_parameters()[0])

    ea.T_Start, ea.t_0, ea.beta, ea.r = -5000, 0, 0.01, 0.01
    errors, cautions = ea.check_valid_parameters()
    assert errors == [] and len(cautions) == 1 and "beta" in cautions[0]


# --- Tie-break: farthest first ---

def test_ties_go_to_the_candidate_farthest_from_what_was_tested():
    # E is empty and every candidate is farther than r from the tested tuple: all score f = 0.
    # The first candidate (earliest t_start) must not win just by coming first.
    tested = np.array([[-4900, -4500]])
    candidates = np.array([[-4700, -4300], [-2500, -1000], [-300, -100]])
    scores = ea.fitness_function(candidates, np.array([]), tested)
    assert np.all(scores == 0)
    chosen = ea.select_test_set(candidates, np.array([]), tested, n=1)
    assert tuple(chosen[0]) == (-300, -100)


def test_search_without_failures_is_not_drawn_to_the_left_edge():
    # With the old tie-break (the first of the tied candidates, which are sorted by t_start),
    # 55 % of the picks had t_start < -4600, where uniform sampling puts 16 %. With E empty
    # every slot now explores, farthest first.
    shares = []
    for seed in range(5):
        _, log, _ = ea.run_simulation(lambda sample: 0, cycles=30, seed=seed)
        shares.append((log.loc[log["cycle"] > 1, "t_start"] < -4600).mean())
    assert np.mean(shares) < 0.3



def test_exploration_pool_must_hold_popsize():
    # While E is empty every slot explores, so the pool must hold popsize tuples.
    ea.popsize, ea.explore_slots, ea.lambda_new, ea.explore_pool = 30, 25, 20, 20
    errors, _ = ea.check_valid_parameters()
    assert len(errors) == 1 and "explore_pool" in errors[0]
    with pytest.raises(ValueError, match="explore_pool"):
        ea.run_simulation(simulation.oracle, cycles=1, seed=1)

    ea.explore_pool = 30  # = popsize: allowed, and the test sets are full, with and without E
    assert ea.check_valid_parameters()[0] == []
    first = ea.step_1(rng=np.random.default_rng(1))
    for verdicts in (np.zeros(30, dtype=int), np.r_[1, np.zeros(29, dtype=int)]):
        archive = ea.record_results(ea.new_archive(), first, verdicts, cycle=1)
        proposed = ea.step_2(archive, first, ea.sigma, rng=np.random.default_rng(2))
        assert proposed.shape[0] == 30


# --- Boundary candidates ---

def test_boundary_candidate_is_the_midpoint_to_the_nearest_pass():
    errors = np.array([[-2000, -500]])
    passes = np.array([[-2000, -400], [-100, -50]])  # the first is the nearest
    assert ea.boundary_candidates(errors, passes).tolist() == [[-2000, -450]]


def test_no_boundary_candidate_once_error_and_pass_are_within_resolution():
    errors = np.array([[-2000, -500]])
    assert ea.boundary_candidates(errors, np.array([[-2000, -499]])).size == 0  # d = 0.75 ms
    assert ea.boundary_candidates(errors, np.empty((0, 2))).size == 0
    assert ea.boundary_candidates(np.empty((0, 2)), np.array([[-2000, -400]])).size == 0


# --- Adaptive sigma ---

def test_sigma_shrinks_after_a_failure_and_grows_after_patience_quiet_cycles():
    ea.patience = 2
    assert ea.update_sigma(50, True, 1) == (pytest.approx(35), 0)  # failure: x 0.7, counter reset
    assert ea.update_sigma(50, False, 0) == (pytest.approx(50), 1)  # one quiet cycle: unchanged
    assert ea.update_sigma(50, False, 1) == (pytest.approx(75), 0)  # patience reached: x 1.5
    assert ea.update_sigma(1.2, True, 0)[0] == pytest.approx(ea.sigma_min)  # clipped at sigma_min


# --- Reflecting bound ---

def test_overshoots_are_mirrored_back_inside_not_clipped_onto_the_bound():
    lo, hi = ea.T_Start - ea.offset, ea.t_0 + ea.offset
    repaired = ea._repair([[lo - 30, -1000], [-1000, hi + 20]])
    assert repaired.tolist() == [[lo + 30, -1000], [-1000, hi - 20]]
    huge = ea._repair([[lo - 10 * (hi - lo), -1000]])  # wider than the domain: clipped
    assert lo <= huge.min() and huge.max() <= hi


# --- Step 1 and the exploration pool: no duplicates, no short sets ---

def test_step_1_holds_popsize_distinct_tuples_even_on_a_tiny_domain():
    # Step 1 is the first popsize points of the run's Halton sequence.
    first = ea.step_1(rng=np.random.default_rng(3))
    assert np.array_equal(first, ea.HaltonSampler(np.random.default_rng(3)).draw(ea.popsize))
    ea.T_Start, ea.t_0, ea.offset = -6, 0, 0  # 21 possible tuples
    for seed in range(20):
        first = ea.step_1(rng=np.random.default_rng(seed))
        assert first.shape == (ea.popsize, 2)
        assert len(np.unique(first, axis=0)) == ea.popsize


def test_exploration_pool_is_topped_up_when_draws_were_already_tested():
    # 15 of the 21 tuples of [-6, 0] are tested, and the pool is a single draw, so most draws
    # are already tested. max_attempts = 1 is the old single-draw pool: many short test sets.
    # With the top-up (default max_attempts) a short set needs every round to miss.
    ea.T_Start, ea.t_0, ea.offset = -6, 0, 0
    ea.popsize, ea.explore_slots, ea.explore_pool = 2, 1, 1
    every = np.array([(a, b) for a in range(-6, 1) for b in range(-6, 1) if b - a >= 1])

    def short_sets(attempts):
        ea.max_attempts = attempts
        short = 0
        for seed in range(30):
            tested = every[np.random.default_rng(seed).permutation(len(every))[:15]]
            archive = ea.record_results(ea.new_archive(), tested, np.zeros(15, dtype=int), cycle=1)
            proposed = ea.step_2(archive, tested[:2], ea.sigma, rng=np.random.default_rng(seed))
            short += proposed.shape[0] < 2
        return short

    assert short_sets(1) >= 15
    assert short_sets(20) <= 1


def test_sigma_bounds_must_fit_together():
    ea.sigma_min, ea.sigma_max_ratio = 600, 0.1  # upper bound 0.1 * 5000 = 500 ms
    errors, _ = ea.check_valid_parameters()
    assert len(errors) == 1 and "sigma_min" in errors[0]



# --- Shared state: the module's seed and generator ---

def test_a_seeded_run_leaves_the_module_generator_untouched():
    state_before = ea.rng.bit_generator.state
    first, _, _ = ea.run_simulation(simulation.oracle, cycles=5, seed=5)
    second, _, _ = ea.run_simulation(simulation.oracle, cycles=5, seed=5)
    assert first.equals(second)
    assert ea.seed is None
    assert ea.rng.bit_generator.state == state_before


def test_reset_module_restores_parameters_E_seed_and_generator():
    from conftest import DEFAULT_PARAMS, DEFAULT_RNG, reset_module

    ea.T_Start, ea.popsize, ea.sigma_max_ratio = -6, 9, 1.0  # as a web-app session can leave them
    ea.error_hit_set = np.array([[-4025, -300]])
    ea.seed_rng(7)
    ea.rng.random(3)
    reset_module()
    assert {name: getattr(ea, name) for name in DEFAULT_PARAMS} == DEFAULT_PARAMS
    assert ea.error_hit_set.size == 0
    assert ea.seed is None
    assert ea.rng.bit_generator.state == DEFAULT_RNG.bit_generator.state



# --- Halton points ---

def test_halton_points_are_reproducible_and_continue_across_draws():
    a = ea.HaltonSampler(np.random.default_rng(4))
    b = ea.HaltonSampler(np.random.default_rng(4))
    unit = a.unit_points(50)
    assert unit.min() >= 0 and unit.max() < 1
    assert np.array_equal(unit, b.unit_points(50))
    split = np.concatenate((a.draw(4), a.draw(4)))  # the sequence continues
    assert np.array_equal(split, b.draw(8))


def test_halton_points_lie_in_the_domain_and_respect_d_min():
    lo, hi = ea.T_Start - ea.offset, ea.t_0 + ea.offset
    points = ea.HaltonSampler(np.random.default_rng(5)).draw(500)
    assert points.min() >= lo and points.max() <= hi
    assert np.all(points[:, ea.t_end_column] - points[:, ea.t_start_column] >= ea.d_min)


def test_halton_points_leave_smaller_gaps_than_random_points_on_average():
    # A single raw set of Halton points can still leave one larger hole (the EA closes those by
    # picking the farthest of a larger pool), so the comparison is over several seeds.
    probe = ea.initialize_samples_array(3000, rng=np.random.default_rng(6))

    def largest_gap(points):
        return ea.distance_to_set(probe, points, weight=ea.w_explore).max()

    halton = [largest_gap(ea.HaltonSampler(np.random.default_rng(seed)).draw(240)) for seed in range(10)]
    random = [largest_gap(ea.initialize_samples_array(240, rng=np.random.default_rng(seed))) for seed in range(10)]
    assert np.mean(halton) < 0.8 * np.mean(random)


# --- Coverage mode and the plain distance for exploration ---

def test_coverage_mode_while_E_is_empty_skips_mutation(monkeypatch):
    calls = []
    real = ea.generate_candidates
    monkeypatch.setattr(ea, "generate_candidates", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = ea.step_1(rng=np.random.default_rng(7))

    passes = ea.record_results(ea.new_archive(), first, np.zeros(ea.popsize, dtype=int), cycle=1)
    proposed = ea.step_2(passes, first, ea.sigma, rng=np.random.default_rng(8))
    assert proposed.shape[0] == ea.popsize and calls == []  # every slot explored

    one_failure = ea.record_results(ea.new_archive(), first, np.r_[1, np.zeros(ea.popsize - 1, dtype=int)], cycle=1)
    proposed = ea.step_2(one_failure, first, ea.sigma, rng=np.random.default_rng(8))
    assert proposed.shape[0] == ea.popsize and calls  # exploitation is back


def test_exploration_uses_the_plain_distance():
    # A differs from the tested tuple by 400 ms in t_start, B by 250 ms in t_end. With w = 0.75,
    # B is farther (187.5 vs 100); with the plain w_explore = 0.5, A is (200 vs 125).
    tested = np.array([[-2000, -1000]])
    a, b = [-1600, -1000], [-2000, -750]
    pool = np.array([a, b])
    assert ea.select_explorers(pool, tested, n=1).tolist() == [a]
    assert ea.select_test_set(pool, np.array([]), tested, n=1).tolist() == [a]  # tie-break
    ea.w_explore = ea.w
    assert ea.select_explorers(pool, tested, n=1).tolist() == [b]
    assert ea.select_test_set(pool, np.array([]), tested, n=1).tolist() == [b]



# --- Coverage cycles once a failure is known ---

def test_every_k_th_cycle_is_a_coverage_cycle_once_E_is_non_empty():
    ea.explore_every = 3
    known = np.array([[-4025, -300]])
    assert ea.is_coverage_cycle(np.empty((0, 2)), cycle=5)  # E empty: always coverage
    assert [ea.is_coverage_cycle(known, cycle) for cycle in range(2, 8)] == [
        False, True, False, False, True, False,
    ]
    assert not ea.is_coverage_cycle(known, cycle=None)  # no cycle given: exploit
    ea.explore_every = 0
    assert not any(ea.is_coverage_cycle(known, cycle) for cycle in range(2, 20))


def test_a_coverage_cycle_skips_exploitation(monkeypatch):
    calls = []
    real = ea.generate_candidates
    monkeypatch.setattr(ea, "generate_candidates", lambda *a, **k: calls.append(1) or real(*a, **k))
    ea.explore_every = 3
    first = ea.step_1(rng=np.random.default_rng(9))
    archive = ea.record_results(ea.new_archive(), first, np.r_[1, np.zeros(ea.popsize - 1, dtype=int)], cycle=1)
    proposed = ea.step_2(archive, first, ea.sigma, rng=np.random.default_rng(1), cycle=3)
    assert proposed.shape[0] == ea.popsize and calls == []  # cycle 3: the whole set explores
    proposed = ea.step_2(archive, first, ea.sigma, rng=np.random.default_rng(1), cycle=4)
    assert proposed.shape[0] == ea.popsize and calls  # cycle 4: exploitation


def test_explore_every_1_is_only_a_caution():
    ea.explore_every = 1
    errors, cautions = ea.check_valid_parameters()
    assert errors == [] and any("explore_every" in message for message in cautions)
