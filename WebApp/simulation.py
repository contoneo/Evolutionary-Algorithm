"""Simulated HiL of the web app: a synthetic oracle, so the EA feedback loop can be exercised
without real hardware, and a benchmark that runs the EA against it for many seeds. It is part of
the web app and isn't meant to be run on its own; results are only shown in the app. Two
rectangular failure regions stand in for the EDR trigger-timing defect. They are hidden from the
EA: every run starts with E = empty, so it has to find them from the verdicts alone. region_1
contains the first example error_hit_set tuple (-4025, -300); region_2 lies far from it, so
finding it as well tests exploration."""

import numpy as np
import pandas as pd

import Evolutionary_Algorithm as ea

# Failure regions: (t_start centre, half width, t_end centre, half width), in ms. A tuple fails
# when it lies strictly inside one of them. Used by both oracle and benchmark.
REGIONS = {
    "region_1": (-4000, 150, -300, 80),
    "region_2": (-4800, 120, -2000, 60),
}

# Points of the domain at which the benchmark measures the largest untested gap.
PROBE_POINTS = 2000


def in_region(t_start, t_end, name):
    """Whether (t_start, t_end) lies inside the region `name`; works on scalars and arrays."""
    start_centre, start_half, end_centre, end_half = REGIONS[name]
    return (np.abs(t_start - start_centre) < start_half) & (np.abs(t_end - end_centre) < end_half)


def oracle(sample):
    t_start, t_end = sample
    return 1 if any(in_region(t_start, t_end, name) for name in REGIONS) else 0


def _wilson(found, n, z=1.96):
    """95 % Wilson interval of the share found / n."""
    if n == 0:
        return np.nan, np.nan
    p = found / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def _ks_distance(values, cdf):
    """Kolmogorov-Smirnov distance between the values and the distribution with this CDF."""
    values = np.sort(np.asarray(values, dtype=float))
    n = values.shape[0]
    expected = cdf(values)
    return max(
        np.max(np.arange(1, n + 1) / n - expected), np.max(expected - np.arange(0, n) / n)
    )


def benchmark(seeds=range(50), cycles=60, on_progress=None):
    """Measure the EA with the current parameters, once per seed, in memory (ea.run_simulation).
    Two runs per seed:
      * coverage: a run in which no test ever fails, so E stays empty and the EA tests the
        domain as evenly as it can (coverage mode); compared with as many uniformly random tests
        per cycle. Measured over the exploitation of a failure, the tests would look uneven on
        purpose, so that part is left to the second run;
      * regions: a run against the simulated HiL (oracle).
    Returns a dict of DataFrames, nothing is printed:
      * coverage: per seed, method ("EA" or "random") and cycle: the largest untested gap (ms,
        largest distance from a probe point of the domain to the nearest tested tuple, in the
        w_explore metric) and the Kolmogorov-Smirnov distance of the tested t_start and t_end
        from those of the uniform domain;
      * coverage_by_cycle: the same, averaged over the seeds;
      * runs: per seed, first-hit cycle of each region (NaN if not found), first failure cycle,
        failures found, termination reason and last cycle;
      * found_by_cycle: per region and cycle, the share of runs that have found it;
      * regions: per region, and for all regions together (the cycle by which the last of them
        was found), runs that found it by the last cycle, the share with its 95 % Wilson
        interval, and the median first-hit cycle.
    on_progress(done, total) is called after each seed."""
    errors, _ = ea.check_valid_parameters()
    if errors:
        raise ValueError(" ".join(errors))
    seeds = list(seeds)
    lo, hi = ea.T_Start - ea.offset, ea.t_0 + ea.offset
    span = hi - lo
    start_cdf = lambda x: 1 - ((hi - np.clip(x, lo, hi)) / span) ** 2
    end_cdf = lambda y: ((np.clip(y, lo, hi) - lo) / span) ** 2
    probe = ea.initialize_samples_array(PROBE_POINTS, rng=np.random.default_rng(0))

    coverage_rows, run_rows = [], []
    for done, seed in enumerate(seeds, start=1):
        no_failures, _, _ = ea.run_simulation(lambda sample: 0, cycles, seed=seed)
        tested = no_failures[no_failures["source"] == "hil"]
        baseline_rng = np.random.default_rng(seed)
        per_cycle = {
            "EA": [
                group[["t_start", "t_end"]].to_numpy()
                for _, group in tested.groupby("cycle", sort=True)
            ],
        }
        per_cycle["random"] = [
            ea.initialize_samples_array(len(points), rng=baseline_rng) for points in per_cycle["EA"]
        ]
        for method, cycle_points in per_cycle.items():
            nearest = np.full(probe.shape[0], np.inf)
            seen = np.empty((0, 2), dtype=int)
            for cycle, points in enumerate(cycle_points, start=1):
                distances = ea._weighted_distance_ms(probe, points, weight=ea.w_explore)
                nearest = np.minimum(nearest, distances.min(axis=1))
                seen = np.concatenate((seen, points), axis=0)
                coverage_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "cycle": cycle,
                        "largest gap (ms)": nearest.max(),
                        "KS t_start": _ks_distance(seen[:, ea.t_start_column], start_cdf),
                        "KS t_end": _ks_distance(seen[:, ea.t_end_column], end_cdf),
                    }
                )

        archive, _, reason = ea.run_simulation(oracle, cycles, seed=seed)
        failures = archive[archive["verdict"] == 1]
        row = {"seed": seed}
        for name in REGIONS:
            hits = failures[in_region(failures["t_start"], failures["t_end"], name)]
            row[f"{name} first hit"] = hits["cycle"].min() if not hits.empty else np.nan
        row["first failure"] = failures["cycle"].min() if not failures.empty else np.nan
        row["failures found"] = len(failures)
        row["termination"] = reason
        row["last cycle"] = int(archive["cycle"].max())
        run_rows.append(row)
        if on_progress is not None:
            on_progress(done, len(seeds))

    coverage = pd.DataFrame(coverage_rows)
    runs = pd.DataFrame(run_rows)
    coverage_by_cycle = (
        coverage.drop(columns="seed").groupby(["method", "cycle"], as_index=False).mean()
    )
    found_rows, region_rows = [], []
    for name in REGIONS:
        first_hit = runs[f"{name} first hit"]
        for cycle in range(1, cycles + 1):
            found_rows.append(
                {"region": name, "cycle": cycle, "share found": (first_hit <= cycle).mean()}
            )
        found = int(first_hit.notna().sum())
        low, high = _wilson(found, len(runs))
        region_rows.append(
            {
                "region": name,
                "runs that found it": f"{found} of {len(runs)}",
                "share": found / len(runs),
                "95 % interval": f"{low:.0%} to {high:.0%}",
                "median first hit (cycle)": first_hit.median(),
            }
        )
    both = runs[[f"{name} first hit" for name in REGIONS]].notna().all(axis=1)
    low, high = _wilson(int(both.sum()), len(runs))
    region_rows.append(
        {
            "region": "all regions",
            "runs that found it": f"{int(both.sum())} of {len(runs)}",
            "share": both.mean(),
            "95 % interval": f"{low:.0%} to {high:.0%}",
            "median first hit (cycle)": runs[[f"{name} first hit" for name in REGIONS]].max(axis=1)[both].median(),
        }
    )
    return {
        "coverage": coverage,
        "coverage_by_cycle": coverage_by_cycle,
        "runs": runs,
        "found_by_cycle": pd.DataFrame(found_rows),
        "regions": pd.DataFrame(region_rows),
    }
