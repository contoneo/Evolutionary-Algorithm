# Evolutionary Algorithm for EDR test samples

### Introduction
This algorithm was presented as a part in the thesis I wrote for my Master of Science (M.Sc.) degree in the “Automotive Software Engineering” program, whose main objective was to provide a testing methodology based on the ISO 26262 standard to address the new regulations taking effect in 2021 for EDRs and DSSADs within the framework of several markets around the world.  EDR (Event Data Recorder) and DSSAD (Data Storage System for Automated Driving) are devices installed in vehicles to record critical driving and crash dynamics data during an accident.

The proposed approach to create test samples assumes the discovery of a malfunction during a test.  Thereafter, it can be used to exploit the flaw, uncovering other related malfunctions, and narrowing down the scope and behavior of the error within the limits of the HiL test.

### Description
In the operation of an EDR, there are different triggers for activating and recording an event, these are defined in government resolutions.  The test under consideration examines whether an activation trigger might prevent the recording of an event, depending on the timing of each trigger.

If some errors are detected during a HiL test cycle for the case described above, the parameters of further tests can be optimized by this Evolutionary Algorithm (hereinafter referred to simply by the acronym **EA**).  Because the optimization needs the HiL verdict of every test it proposes, the **EA** and the HiL test run as one interactive loop: propose a test set, run it, record pass/fail, propose the next set (see "Algorithm" below).

![](EDR_activity_state.png)

*EDR test case for which the **EA** is made*

#### Population

A test set for this test case (**EA**'s output) is a collection of tuples setting the start and the end of the so called "trivial event" in the way
$$(t_{\text{start}}, t_{\text{End}})$$
where
* $t_{\text{End}} - t_{\text{start}} \ge d_{\min}$ (a minimum duration; zero-length events are never produced) and $t_{\text{End}} \in T$.
* $t_{\text{start}}$ may lie before $T_{\text{Start}}$: only $t_{\text{End}}$ is required to be inside $T$, and it is only *penalised* (not forbidden) for lying outside it, by $V_2$ below.
* $T = [T_{\text{Start}}, t_0]$: Pre-crash recording time established in the regulation. $t_0$ is the instant when the recording algorithm starts, normally set to zero.
* Candidate tuples are clipped to $[T_{\text{Start}} - \text{offset},\ t_0 + \text{offset}]$, so `offset` bounds how far a search can drift from the test case, not just the random samples of Step 1.

All times are given in milliseconds.

#### Fitness Orientation

The optimization made by the **EA** works according to three software testing principles, kept as three separate, normalised terms:
* **Defect clustering** ($V_1$): most software bugs are concentrated within a small number of complex or heavily used modules, so the search is attracted towards known errors.
* **Staying within the regulation window** ($V_2$): the main purpose of the HiL test is to verify the EDR within the pre-crash recording time $T$, so $t_{\text{End}}$ outside $T$ is penalised.
* **The Pesticide Paradox** ($V_3$): repeatedly running the same tests will eventually stop finding new bugs, so the search is repelled from everything already tested — this is what keeps the EA exploring instead of only re-testing known failures.

Let $|T| = t_0 - T_{\text{Start}}$ (5000 ms by default). For a tuple $x=(t_{\text{start}}, t_{\text{End}})$:

$$f(x) = V_1(x) + \gamma \cdot V_2(x) + \beta \cdot V_3(x), \qquad \text{lower is better}$$

* $\displaystyle V_1(x) = \frac{(1-w)\cdot\min\limits_{i \in E}\big|t_{\text{start}} - E_i(t_{\text{start}})\big| \;+\; w\cdot\min\limits_{j \in E}\big|t_{\text{End}} - E_j(t_{\text{End}})\big|}{|T|}$
  * $E$: set of test tuples that have led to an error (error_hit_set). $V_1 = 0$ if $E$ is empty.
  * $t_{\text{start}}$ and $t_{\text{End}}$ are chosen **independently**: the closest known start time and the closest known end time may come from *different* tuples of $E$, so the algorithm can also search cross-combinations of known start and end times, not only tuples that exactly repeat $E$. A cross-combination that has never been tested gives $V_1 = 0$ (and $V_3 = 0$, see below): that is a genuinely new test, not a re-test.
  * $w \in (0.5, 1)$ replaces the old $a$, $b$ and the $/1000$ scale: $t_{\text{End}}$ is more important than $t_{\text{start}}$, so $w > 0.5$. $w$ is also used, the same way, in the whole-tuple distance $d$ used by $V_3$ below.
* $\displaystyle V_2(x) = \frac{\max(0,\ T_{\text{Start}} - t_{\text{End}},\ t_{\text{End}} - t_0)}{|T|}$ — the distance from $t_{\text{End}}$ to the nearest bound of $T$, 0 while $t_{\text{End}} \in T$. It stays a **soft penalty** on an otherwise unconstrained $t_{\text{start}}$/$t_{\text{End}}$ pair, not a hard constraint.
* $\displaystyle V_3(x) = \max\Big(0,\ 1 - \frac{d(x, y^*)}{r}\Big)$, where $y^* = \arg\min_{y \in A} d(x, y)$.
  * $A$: archive of every tuple tested so far, pass or fail ($E \subset A$). $V_3 = 0$ if $A$ is empty.
  * $\displaystyle d(x, y) = \frac{(1-w)\cdot|x_{\text{start}} - y_{\text{start}}| + w\cdot|x_{\text{End}} - y_{\text{End}}|}{|T|}$: the same weighted distance as $V_1$, but between two **whole tuples** (not per-coordinate). A per-coordinate novelty term would forbid ever reusing a single known start or end time; the whole-tuple distance only forbids reusing both at once.
  * $r$: novelty radius. $V_3$ is highest ($=1$) at distance 0 from a tested tuple and reaches 0 at distance $r$.
* **Ring condition:** because $V_1$ and $V_3$ pull in opposite directions near a known error, the optimum of $f$ sits on a ring of radius $\approx r$ around a tested error only when $\beta > r$. Choose $\beta$ accordingly.
* With $E = \varnothing$, $V_1 = 0$ and $f$ reduces to pure novelty search (exploration). As $E$ fills up, $f$ balances exploitation (toward $E$) against exploration (away from $A$).

The best scored tuple by $f$ is the lowest scored.

#### Variation Drive
The **EA** uses **linear-ranking roulette**: it keeps the "wheel" idea of Roulette Wheel Selection (RWS), but selects by rank instead of by the raw (and scale-dependent) value of $f$, so it stays well-behaved when $f = 0$ or when scores are close together.

Parents are ranked by $f$: rank $0$ is the worst (highest $f$), rank $n-1$ the best (lowest $f$), where $n$ is the size of the parent pool. The parent pool is the **last test set $\cup\ E$**, so every known error cluster stays available for selection even when it fell out of the last test set.

$$p_i = \frac{2-s}{n} + \frac{2\cdot \text{rank}_i\cdot(s-1)}{n\cdot(n-1)}, \qquad \sum_{i=1}^{n} p_i = 1$$

* $s \in [1, 2]$: selection pressure (default 1.5). $s=1$ gives every rank the same probability; $s=2$ gives the worst rank a probability of 0.
* As in the original RWS, a parent $k$ is selected when $\sum_{i=1}^{k-1} p_i \le \text{rand} < \sum_{i=1}^{k} p_i$, repeated `lambda_new` times.

#### Mutation
The **EA** does not use recombination: the selected parent tuples are mutated directly, with "normal mutation" on each parameter independently: $x_i = x_i + N(0, \sigma)$, rounded to whole milliseconds.

* **Adaptive $\sigma$**, driven by HiL feedback: $\sigma \leftarrow \sigma \cdot 0.7$ after a cycle that found a new failure (exploit locally), $\sigma \leftarrow \sigma \cdot 1.5$ after `patience` cycles without one (explore further), clipped to $[\sigma_{\min}, \sigma_{\max}]$.
* **Random immigrants:** a fraction `immigrant_rate` of each candidate batch is drawn fresh from the Step 1 generator instead of from mutation, so the search keeps exploring even while $\sigma$ is small.
* **Repair:** candidates are clipped into $[T_{\text{Start}}-\text{offset},\ t_0+\text{offset}]$, swapped if $t_{\text{start}} > t_{\text{End}}$, and stretched to the minimum duration $d_{\min}$ if too short.
* **Duplicates are removed:** a candidate repeated within the batch, or already tested ($A$), is discarded. Generation is retried (up to `max_attempts` rounds) until enough new candidates exist, so a batch that only produces duplicates does not stall a cycle.

#### Test Set Selection
The **EA**'s output is a **test set**, run on the HiL as one batch, so survivor selection is **greedy set selection** over the candidates only (comma selection: a tuple that has already been tested is never proposed again, unlike the old $(\mu+\lambda)$ elitism):

Repeat `popsize` times: score every remaining candidate with $f$ against $A \cup \text{chosen}$ (the archive plus the tuples already picked for this set), take the lowest, add it to `chosen`. Because $V_3$ then also penalises proximity to the tuples just chosen, the set spreads out over the error clusters instead of collapsing onto one of them.

### Algorithm

One EA cycle corresponds to one HiL test run:

1. **Step 1 (first test set):** `initialize_samples_array()` creates `popsize` random tuples within $T$ extended by `offset`. (The CLI demo and the tests use a fixed example population from `step_1()` instead, for reproducibility.)
2. **Run** the test set on the HiL and get one verdict per tuple: fail (error found) or pass.
3. **Record:** append the tuples with their verdict and cycle number to the archive $A$. Failures extend $E$; passes extend the "known good" region used by boundary candidates.
4. **Adapt $\sigma$** from whether the cycle found a new failure.
5. **Boundary candidates:** for each error $e$, find its nearest passed tuple $p$. If their distance is wider than `resolution_ms`, add the rounded midpoint $(e+p)/2$ as a candidate; it competes like any other candidate. Because passed tuples repel through $V_3$ too, the search fills in the region between errors and passes — where the failure boundary lies.
6. **Step 2 (propose):** select parents (Variation Drive), generate candidates (Mutation) and boundary candidates (step 5), then pick the next test set (Test Set Selection). Back to step 2.

**Termination:** stop when no error/pass pair is wider than `resolution_ms`, or after `max_cycles`, whichever comes first. The run reports which one triggered the stop.

### Reproducibility

* A single `np.random.Generator` (`rng`, from `np.random.default_rng(seed)`) is threaded through every random function; the same `seed` reproduces the same run bit-for-bit.
* The archive (every tuple tested, its verdict and its cycle) is persisted as `archive.csv`.
* Each cycle is logged to `runs/<timestamp>_log.csv`, with one row per tuple: `cycle, t_start, t_End, V1, V2, V3, f, sigma, seed, verdict`.

### Parameters

All parameters are module variables in [Evolutionary_Algorithm.py](Evolutionary_Algorithm.py) and can be changed in the web app.

| Parameter | Default | Meaning |
|---|---|---|
| `T_Start` | -5000 | Start of the pre-crash recording time $T$ (ms) |
| `t_0` | 0 | End of the pre-crash recording time $T$ (ms) |
| `offset` | 10 | Extension of $T$ for Step 1 and random immigrant samples (ms) |
| `popsize` | 4 | Test set size |
| `lambda_new` | 20 | Candidates generated per cycle (before greedy test-set selection) |
| `w` | 0.75 | Weight of $t_{\text{End}}$ vs $t_{\text{start}}$ in $V_1$ and $d$ |
| `gamma` | 10 | Weight of $V_2$ in $f$ |
| `r` | 0.01 | Novelty radius for $V_3$ (normalised by $\lvert T \rvert$; $\approx$ 50 ms) |
| `beta` | 0.05 | Weight of $V_3$ in $f$; must be $>$ `r` |
| `s` | 1.5 | Linear-ranking selection pressure, in $[1, 2]$ |
| `sigma` | 50 | Initial standard deviation of the normal mutation (ms) |
| `sigma_min`, `sigma_max` | 1, $\lvert T \rvert / 10$ | Bounds for adaptive $\sigma$ |
| `patience` | 2 | Cycles without a new failure before $\sigma$ grows |
| `immigrant_rate` | 0.2 | Share of each candidate batch drawn fresh instead of mutated |
| `d_min` | 1 | Minimum event duration $t_{\text{End}} - t_{\text{start}}$ (ms) |
| `resolution_ms` | 1 | Boundary resolution and stop criterion (ms) |
| `max_cycles` | 4 | HiL cycle limit |
| `seed` | `None` | RNG seed; `None` means a fresh, non-reproducible run |
| `error_hit_set` | empty | $E$: manually known failing tuples, merged into the archive as fail rows at cycle 0 |

### How to run

Requirements: Python 3 (the [mise.toml](mise.toml) sets Python 3.13 and a `.venv`) and the packages in [requirements.txt](requirements.txt).

```sh
pip install -r requirements.txt
```

**Command line:** runs a synthetic-oracle simulation (`simulation.py`), so the full HiL feedback loop (propose → "test" → record → propose) can be seen without real hardware. It starts from `step_1()`'s fixed example population, prints the resulting archive, and reports the termination reason.

```sh
python Evolutionary_Algorithm.py
```

**Web app:** a Streamlit interface that drives one real HiL loop interactively — you provide the pass/fail verdicts.

```sh
streamlit run WebApp/layout.py
```

1. Set the domain, fitness weight, mutation and population parameters (or keep the defaults).
2. Optionally enter a manually known `error_hit_set`, one tuple per line in the format `t_Start, t_End` (e.g. `-4025, -300`), and press **Enter `error_hit_set`**.
3. Press **Step 1: create first test set**.
4. Run the test set on the HiL, tick **error found** for every tuple that failed in the table, and press **Record HiL results**.
5. Press **Propose next test set (Step 2)** and repeat from step 4.
6. Download the archive at any point with **Download archive CSV**; reload it in a later session with **Upload archive CSV** + **Load uploaded archive**.

**Tests:**

```sh
pytest
```

### Notes

* `Evolutionary_Algorithm.ipynb` documents the pre-rewrite version of the algorithm and has not been updated to match this README; it is kept for reference only.
