import numpy as np
import pytest

import Evolutionary_Algorithm as ea
from WebApp import simulation


def test_oracle_fails_exactly_inside_the_regions():
    for name, (start_centre, _, end_centre, _) in simulation.REGIONS.items():
        assert simulation.oracle((start_centre, end_centre)) == 1
        assert simulation.in_region(start_centre, end_centre, name)
    assert simulation.oracle((-2500, -1000)) == 0


def test_benchmark_returns_its_tables_and_leaves_no_trace():
    state_before = ea.rng.bit_generator.state
    result = simulation.benchmark(range(2), cycles=20)

    assert len(result["runs"]) == 2
    assert len(result["coverage"]) == 2 * 2 * 20  # seeds x methods x cycles
    assert set(result["coverage_by_cycle"]["method"]) == {"EA", "random"}
    assert len(result["found_by_cycle"]) == len(simulation.REGIONS) * 20
    assert list(result["regions"]["region"]) == [*simulation.REGIONS, "all regions"]
    assert {"first failure", "failures found", "termination", "last cycle"} <= set(result["runs"].columns)

    assert ea.seed is None and ea.rng.bit_generator.state == state_before


def test_benchmark_ea_covers_the_domain_better_than_random_tests():
    coverage = simulation.benchmark(range(3), cycles=30)["coverage_by_cycle"]
    last = coverage[coverage["cycle"] == 30].set_index("method")
    assert last.loc["EA", "largest gap (ms)"] < last.loc["random", "largest gap (ms)"]


def test_benchmark_refuses_invalid_parameters():
    ea.explore_pool = ea.popsize - 1
    with pytest.raises(ValueError, match="explore_pool"):
        simulation.benchmark(range(1), cycles=5)


def test_ks_distance_of_a_perfect_sample_is_small():
    lo, hi = ea.T_Start - ea.offset, ea.t_0 + ea.offset
    end_cdf = lambda y: ((np.clip(y, lo, hi) - lo) / (hi - lo)) ** 2
    uniform = ea.initialize_samples_array(20000, rng=np.random.default_rng(1))
    assert simulation._ks_distance(uniform[:, ea.t_end_column], end_cdf) < 0.02
    assert simulation._ks_distance(np.full(100, hi), end_cdf) > 0.9
