"""Synthetic HiL oracle, so the EA feedback loop (A2/A3) can be exercised without real
hardware. Two rectangular failure regions stand in for the EDR trigger-timing defect."""

import Evolutionary_Algorithm as ea


def oracle(sample):
    t_start, t_end = sample
    region_1 = abs(t_start + 4000) < 150 and abs(t_end + 300) < 80
    region_2 = abs(t_start + 1800) < 120 and abs(t_end + 100) < 60
    return 1 if (region_1 or region_2) else 0


def main():
    # The rectangular regions are small relative to T (filled areas, not single points), so a
    # realistic demo budget is well above ea.max_cycles (the algorithm's own per-cycle default):
    # a handful of cycles only explores, it takes dozens more to land inside a ~300x160 ms box
    # by novelty search alone, since nothing attracts the search until the first failure.
    demo_cycles = 60
    archive, log, reason = ea.run_simulation(
        oracle,
        max_cycles=demo_cycles,
        seed=42,
        initial_test_set=ea.step_1(),
    )
    print(archive)
    cycles_run = int(archive["cycle"].max()) if not archive.empty else 0
    fails = archive[archive["verdict"] == 1]
    first_fail_cycle = int(fails["cycle"].min()) if not fails.empty else None
    duplicates = archive.duplicated(subset=["t_start", "t_End"]).sum()
    print(f"Cycles run: {cycles_run}")
    print(f"First failure found at cycle: {first_fail_cycle}")
    print(f"Failures found: {len(fails)}")
    print(f"Duplicate tuples tested: {duplicates}")
    print(f"Termination reason: {reason}")
    return archive, log, reason


if __name__ == "__main__":
    main()
