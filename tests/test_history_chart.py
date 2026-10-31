import numpy as np

import Evolutionary_Algorithm as ea
from WebApp import history_chart


def _archive():
    """A small run: cycle 1 tests two tuples (one fails), cycle 2 tests one (a pass); a manual
    failure is added at cycle 2, and the cycle-1 pass is later marked as failed (override)."""
    archive = ea.record_results(ea.new_archive(), [[-2000, -500], [-1000, -100]], [1, 0], cycle=1)
    archive = ea.record_results(archive, [[-3000, -700]], [0], cycle=2)
    archive = ea.merge_manual_error_hit_set(archive, [[-4025, -300]], cycle=2)
    return ea.merge_manual_error_hit_set(archive, [[-1000, -100]], cycle=3)


def test_run_points_label_every_kind_of_result():
    points = history_chart.run_points(_archive())
    result = {(p.t_start, p.t_end): p.result for p in points.itertuples()}
    assert result == {
        (-2000, -500): "fail",
        (-1000, -100): "override",
        (-3000, -700): "pass",
        (-4025, -300): "manual failure",
    }


def test_cycle_metrics_count_hil_verdicts_of_each_cycle_only():
    points = history_chart.run_points(_archive())
    metrics = history_chart.cycle_metrics(points, {1: 50.0, 2: 35.0})
    by_cycle = metrics.set_index("cycle")
    assert by_cycle.loc[1, "failures per test"] == 1.0  # the override moved to cycle 3
    assert by_cycle.loc[2, "failures per test"] == 0.0  # the manual failure isn't a test
    assert np.isnan(by_cycle.loc[3, "failures per test"])  # no HiL test at cycle 3
    assert by_cycle.loc[2, "sigma"] == 35.0


def test_charts_build_for_a_run():
    points = history_chart.run_points(_archive())
    overview = history_chart.overview_chart(points, -5000, 0, 10, 1, last_cycle=3)
    spec = overview.to_dict()
    assert spec["layer"]  # domain, window, passes and failures
    assert "[-5010, 10]" in str(spec)  # the title shows the search domain, T plus offset
    for chart in history_chart.metrics_charts(history_chart.cycle_metrics(points, {})):
        assert chart.to_dict()["mark"]["type"] == "line"
