# HiL loop visualization: review of Option 3

Review of a visual feature for the "HiL loop" section of [WebApp/layout.py](WebApp/layout.py). It shows how the test sets proposed by "Propose next test set (Step 2)" develop over the cycles.

## Background

**Option 3** (chosen from four alternatives): a layered Altair chart of the 2-D search space (`t_Start` × `t_End`), drawn with `st.altair_chart`. Altair is already installed with Streamlit, so `requirements.txt` does not change.

Requirements given for the feature:

1. **Not a real scenario.** The EDR trigger-timing test is an illustration of how an EA can suggest test samples; it will never run on a real HiL.
2. **Show the README goal:** "it can be used to exploit the flaw, uncovering other related malfunctions, and narrowing down the scope and behavior of the error within the limits of the HiL test."
3. **Keep the history** of proposals for a given seed, starting from the same first test set. Pressing "Step 1: create first test set" must **not** drop it.

## What the requirements change

### 1. No real HiL: the simulation oracle stands in for it

- Pass/fail results come from `simulation.oracle`. The "error found" checkboxes are pre-filled when a set is proposed, and the user still presses **Record**, so the HiL loop stays visible.
- A run is fully reproducible: same seed, same parameters and same known error give the same history. With manual verdicts this did not hold, because two runs matched only if the same failures were ticked.
- The true failure regions are known, so the chart can draw them, and the viewer can judge how good the proposals are.
- A **"Run N cycles"** button is needed: mapping a region takes dozens of cycles (see the experiment below), too many to click one by one.
- For the chart to draw the regions, [simulation.py](simulation.py) needs one small change: move the two rectangles into a `REGIONS` constant that `oracle` reads. The oracle's results stay the same, and no test uses it (the tests define their own oracles).

### 2. The README goal, checked by experiment

**Setup:** `ea.run_simulation(simulation.oracle, max_cycles=60, seed=…)` with the default parameters (`popsize=4`, `lambda_new=20`, `beta=0.05`, `r=0.01`, `immigrant_rate=0.2`, `sigma=50`). The known error is the README example `error_hit_set = [(-4025, -300)]`, which lies inside region 1. Seeds: 1, 7, 42, 123.

Oracle regions (`simulation.py`):

| Region | t_Start | t_End |
|---|---|---|
| 1 | (-4150, -3850) | (-380, -220) |
| 2 | (-1920, -1680) | (-160, -40) |

**Results:**

| Goal | Result | How the visual can show it |
|---|---|---|
| **Exploit the flaw** | ✅ Region 1 is hit in cycle 2 in every run; 188–197 of 240 tests fail (about 80%). Without the known error, seeds 1 and 42 found nothing in 60 cycles, seed 123 first hit at cycle 48, seed 7 at cycle 3. | Points crowd around the known error (drawn as a star). Plot hit rate per cycle; cycle 1, which is random, is the baseline. |
| **Narrow down scope and behaviour** | ✅ The box around the found failures closes in on the true region. Seed 1: t_Start [-4149, -3851] × t_End [-379, -221], against a true region of (-4150, -3850) × (-380, -220). | Draw the box around found failures per region (solid) over the true region (dashed). Plot "share of region mapped" per cycle and list the found ranges, e.g. "t_End range is half as wide as t_Start's". |
| **Uncover related malfunctions** | ❌ Region 2 was **never** found with the default parameters (0 of 4 runs). | The chart would show this gap honestly. |

**Region 2 with tuned parameters** (same setup, known error given, 60 cycles):

| beta | r | immigrant_rate | Runs that found region 2 |
|---|---|---|---|
| 0.05 | 0.01 | 0.2 | 0 / 4 |
| 0.05 | 0.01 | 0.5 | 0 / 4 |
| 0.3 | 0.05 | 0.2 | 1 / 4 (seed 7, cycle 38) |
| 0.5 | 0.05 | 0.3 | 0 / 4 |
| 1.0 | 0.1 | 0.3 | 1 / 4 (seed 42, cycle 41) |
| 0.5 | 0.1 | 0.5 | 1 / 4 (seed 1, cycle 10) |

The third part of the goal is therefore a limit of the algorithm, not of the visual. Once failures exist, V1 pulls every selection toward the known errors, and sigma keeps shrinking (×0.7 after every cycle with a failure). Random immigrants are generated but almost never win the greedy selection, because their V1 (distance to the errors) is far larger than the novelty penalty `beta · V3` on candidates near region 1.

Options:

- **(a) Accept it**, and show only exploit and narrow-down.
- **(b) Pick demo parameters and a seed where it works**, e.g. `beta=0.5, r=0.1, immigrant_rate=0.5`, seed 1, which finds region 2 at cycle 10. Fragile and somewhat cherry-picked.
- **(c) A separate algorithm change**, e.g. reserving one slot per test set for pure exploration (novelty only, without V1). This goes beyond keeping changes small, and should be its own task.

**Update: option (c) implemented** as the parameter `explore_slots` (default 1). In each test set, that many slots are filled by greedy farthest-point picks from a fresh random pool of `lambda_new` tuples, with no V1 (`select_explorers` in `Evolutionary_Algorithm.py`); `explore_slots = 0` reproduces the previous behaviour bit for bit. Same experiment, 8 seeds (1, 7, 42, 123, 5, 99, 2024, 314), known error given, 60 cycles:

| explore_slots | Region 1 found | Region 1 mapped (avg) | Region 2 found | Failures (avg of 240 tests) |
|---|---|---|---|---|
| 0 | 8 / 8 | 79% | 0 / 8 | 191 |
| 1 | 8 / 8 | 69% | 1 / 8 (cycle 22) | 131 |
| 2 | 8 / 8 | 75% | 3 / 8 (cycles 28, 38, 54) | 78 |

With `explore_slots = 1` and 200 cycles, region 2 was found in 5 of 8 runs (cycles 22–196). A 10× larger random pool did not help (4 of 8). The limit is the test budget, not the selection: region 2 covers about 0.2% of the feasible domain, so even perfect space-filling needs several hundred exploration tests to hit it reliably. "Region 1 mapped" is the area of the box around found failures divided by the true region's area.

For the visual this means: "uncover related malfunctions" can now be shown, but only in long runs (e.g. via "Run N cycles") or for seeds where it happens early, such as seed 2024, which finds region 2 at cycle 22 with `explore_slots = 1`.

### 3. History is kept across Step 1

- Each Step 1 starts a **new run**. Earlier runs stay in a list `st.session_state.runs`, and each entry holds the seed, a snapshot of the parameters, the known error and that run's archive.
- This needs its own store: Step 1 must still reset the algorithm's archive (`ea.new_archive()`), otherwise the runs would mix and `step_2`'s fitness would be wrong.
- The demo gets stronger, because runs can be compared, e.g. "with known error vs. without" (the experiment shows a clear difference) or seed against seed.
- Add a **"Clear history"** button, so dropping the history is a deliberate choice.

## Issues in the current button flow

- **Step 1 doesn't reseed.** It uses the session's random generator in whatever state it is in, so pressing Step 1 twice with the same seed gives two different first sets unless "Apply seed" is pressed in between. Also, `st.session_state.rng` starts as the module-level `ea.rng`, which every browser session shares.
- **Recording twice:** pressing "Record HiL results" twice stores the same set again under cycle k+1, which gives duplicate points and a wrong cycle count.
- **Proposing twice:** pressing "Propose next test set" twice without recording throws away a proposal but still uses up random numbers, so the run no longer matches the same seed's run.
- **Parameters can change mid-run:** they are module-level `ea.<param>` globals edited live in the UI. A run must save them when it starts, and warn if they change.

## Revised Option 3

### Chart (Altair)

- **Background:** the domain box `[T_Start, t_0]` and the shaded area where `t_End − t_Start < d_min`.
- **Ground truth:** true failure regions as dashed rectangles. **Estimate:** the box around found failures per region, solid.
- **Points** coloured by cycle (sequential scale), with shape for the verdict: ✕ fail, ○ pass, ★ known error (`cycle = 0`), ▲ proposal not yet tested.
- **Two panels: overview and zoom.** The domain is about 5000 ms wide and the regions are about 300 × 160 ms, so in the overview all the action collapses into one blob. The second panel zooms automatically on the failure regions.
- **Cycle slider** (1…k): controls both the chart and the metrics; older cycles are faded.
- **Tooltip:** `t_Start`, `t_End`, cycle, verdict.

### Metrics over cycles

- hit rate (cycle 1 = random baseline)
- regions found
- share mapped per region (box around found failures / true region area)
- sigma

### Runs

- A run selector, with an option to show two runs side by side.
- A caption with the run's seed and parameter snapshot.

### Button flow

- **Step 1** reseeds from the seed field and starts a new run.
- **Record** and **Propose** take turns, so a set can't be recorded twice and no proposal is silently thrown away.
- **"Run N cycles"** follows the same order as `run_simulation`: run the set on the oracle, record, update sigma, stop if `boundary_resolved`, otherwise call `step_2`. A web run with seed X then matches `run_simulation(seed=X)`, which is a useful consistency check.
- **"Clear history"** drops all saved runs.

### Files touched

- [WebApp/layout.py](WebApp/layout.py): most of the work.
- [simulation.py](simulation.py): the small `REGIONS` change.
- [Evolutionary_Algorithm.py](Evolutionary_Algorithm.py): unchanged.

## Open points

1. **Region 2:** option (c) is implemented (see above). Still open: whether the demo needs longer runs or a chosen seed to show it.
2. **Checkboxes:** keep them editable as a manual override after the oracle pre-fills them, or make the oracle results read-only? *Implemented: editable override; the oracle can be switched off with the "Simulated HiL" toggle.*
3. **History across Step 1:** is "earlier runs stay available for comparison" the intended meaning? *Implemented: earlier runs stay selectable one at a time; showing two runs side by side was left out.*

## Implementation status

Revised Option 3 is implemented in [WebApp/history_chart.py](WebApp/history_chart.py) (charts and metrics) and [WebApp/layout.py](WebApp/layout.py) (state, buttons, "Test set development" section).

The failure-region parts were implemented and then dropped again: the true and found region boxes, the per-region zoom panels, the "regions found" / "share mapped" metrics, and `REGIONS` / `region_of` in `simulation.py`. What remains is the search-space chart with the cycle slider, the run history, the simulated HiL with "Run N cycles", and the hit-rate and sigma metrics. Since this review, the manual "Record HiL results" button was removed (verdicts are saved as they are ticked), so "Recording twice" no longer applies.
