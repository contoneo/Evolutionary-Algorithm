import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import logging

import numpy as np
import pandas as pd
import streamlit as st

import Evolutionary_Algorithm as ea
from WebApp import history_chart, simulation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Parameters saved with each run at Step 1, so a run can be traced back to its settings.
RUN_PARAMS = (
    "T_Start", "t_0", "offset", "w", "gamma", "r", "beta", "s", "sigma", "patience",
    "immigrant_rate", "d_min", "resolution_ms", "popsize", "lambda_new", "explore_slots",
    "explore_pool", "sigma_min", "sigma_max_ratio", "max_attempts", "w_explore",
    "explore_every",
)


@st.cache_resource
def parameter_defaults():
    """The module defaults of the WebApp parameters, read once per server process (in memory),
    before any session changed the ea.<param> globals: every new browser session starts from
    them, so a browser restart is a fresh start."""
    return {name: getattr(ea, name) for name in RUN_PARAMS}


def init_state():
    """Per-session state: the archive, current test set, sigma and cycle are stateful and
    belong to this HiL session, unlike the module-level parameters below (w, gamma, popsize,
    ...), which stay plain `ea.<param>` globals shared by every session, same as before."""
    st.session_state.setdefault("archive", ea.new_archive())
    st.session_state.setdefault("test_set", np.array([]))
    st.session_state.setdefault("test_set_version", 0)
    st.session_state.setdefault("current_rows", [])
    st.session_state.setdefault("sigma", ea.sigma)
    st.session_state.setdefault("cycle", 0)
    st.session_state.setdefault("cycles_without_failure", 0)
    # True once Step 2 found no untested candidate: the run is finished (reset by Step 1).
    st.session_state.setdefault("no_more_candidates", False)
    # A fresh, unseeded generator per session; Step 1 reseeds it when a seed is given.
    st.session_state.setdefault("rng", np.random.default_rng())
    # Every run started by Step 1 (seed, parameters, archive), kept until "Clear history".
    st.session_state.setdefault("runs", [])


def test_set_dataframe(test_set, verdicts):
    df = pd.DataFrame(test_set, columns=("t_start", "t_end"))
    df["error found"] = verdicts
    return df


def initial_verdicts(test_set):
    """With the simulated HiL on, the oracle pre-fills the verdicts (more failures can still be
    marked by hand); otherwise they default to "no error found"."""
    if st.session_state.get("simulated_hil", True):
        return np.array([bool(simulation.oracle(tuple(row))) for row in test_set], dtype=bool)
    return np.zeros(len(test_set), dtype=bool)


def record_test_set(test_set, cycle):
    """Make test_set the current one and record it into the archive immediately, with its
    initial verdicts, and remember its archive row labels so later checkbox edits
    (apply_editor_edits) can be written straight back into those same rows."""
    st.session_state.test_set = test_set
    st.session_state.test_set_version += 1
    st.session_state.archive = ea.record_results(
        st.session_state.archive, test_set, initial_verdicts(test_set).astype(int), cycle
    )
    st.session_state.current_rows = st.session_state.archive.tail(len(test_set)).index.tolist()


def current_verdicts():
    """The current test set's verdicts, read from its archive rows: the archive is the only
    record of E, so the table always shows what E holds."""
    archive = st.session_state.archive
    return archive.loc[st.session_state.current_rows, "verdict"].astype(bool).to_numpy()


def failure_in_current_cycle():
    """Whether E gained a failure in the current cycle: a failing tuple of the current test
    set, or a failure added by hand during this cycle."""
    archive = st.session_state.archive
    return bool(((archive["cycle"] == st.session_state.cycle) & (archive["verdict"] == 1)).any())


def propose_next_test_set():
    """README "Algorithm" items 4 and 5, in the same order as ea.run_simulation: adapt sigma from
    the failures of the current cycle, propose the next set (Step 2), record it. Returns False,
    and marks the run as finished, when Step 2 finds no untested candidate; state changes only
    once step_2 succeeded."""
    has_failure = failure_in_current_cycle()
    new_sigma, new_cycles_without_failure = ea.update_sigma(
        st.session_state.sigma, has_failure, st.session_state.cycles_without_failure
    )
    next_test_set = ea.step_2(
        st.session_state.archive, st.session_state.test_set, new_sigma,
        rng=st.session_state.rng, sampler=st.session_state.sampler,
        cycle=st.session_state.cycle + 1,
    )
    if next_test_set.shape[0] == 0:
        st.session_state.no_more_candidates = True
        return False
    st.session_state.sigma = new_sigma
    st.session_state.cycles_without_failure = new_cycles_without_failure
    st.session_state.cycle += 1
    record_test_set(next_test_set, st.session_state.cycle)
    st.session_state.runs[-1]["sigma"][st.session_state.cycle] = new_sigma
    return True


def run_label(index, run):
    seed = "random" if run["seed"] is None else run["seed"]
    cycles = int(run["archive"]["cycle"].max())
    failures = len(ea.archive_error_set(run["archive"]))
    return f"Run {index + 1}: seed {seed}, {failures} failure(s) in E, {cycles} cycle(s)"


def apply_editor_edits(editor_key):
    """Write the editor's checkbox edits straight into the archive rows of the current test
    set, so a newly checked "error found" joins E at once. Failures can only be added: an
    unchecked failure is refused, and the editor is redrawn (new key) from the archive."""
    edits = st.session_state[editor_key].get("edited_rows", {})
    archive = st.session_state.archive
    refused = False
    for row, changes in edits.items():
        if "error found" not in changes:
            continue
        label = st.session_state.current_rows[int(row)]
        if changes["error found"]:
            archive.loc[label, "verdict"] = 1
        elif archive.loc[label, "verdict"] == 1:
            refused = True
    st.session_state.archive = archive
    if refused:
        st.session_state.test_set_version += 1
        st.session_state.table_message = "A failure can't be removed from E; the mark was kept."


def add_failures_to_E():
    """Callback of "Add to E": merge the typed tuples into the archive at the current cycle
    (ea.merge_manual_error_hit_set) and clear the input. Messages are shown on the next draw."""
    try:
        tuples = ea.parse_error_hit_set(st.session_state.e_input)
    except ValueError as e:
        st.session_state.e_message = ("error", f"Invalid `error_hit_set`: {e}")
        return
    if tuples.size == 0:
        st.session_state.e_message = ("error", "Enter at least one tuple `t_start, t_end`.")
        return
    archive = st.session_state.archive
    before = len(ea.archive_error_set(archive))
    st.session_state.archive = ea.merge_manual_error_hit_set(
        archive, tuples, cycle=st.session_state.cycle
    )
    added = len(ea.archive_error_set(st.session_state.archive)) - before
    message = f"{added} failure(s) added to E at cycle {st.session_state.cycle}."
    if added < len(tuples):
        message += f" {len(tuples) - added} already in E."
    st.session_state.e_message = ("success", message)
    st.session_state.e_input = ""


def render_error_hit_set():
    """E is interactive: failures marked in the current test set and failures typed in here
    are merged into the archive at the current cycle. Before the first Step 1 that is cycle 0:
    the failures are kept and carried into the run Step 1 starts. Failures can be added, never
    removed."""
    st.divider()
    st.subheader("Error hit set E")
    col_add, col_list = st.columns(2)
    run_started = st.session_state.test_set.size > 0
    col_add.text_area(
        # Markdown collapses single newlines; two trailing spaces force a line break.
        "Add failures: one tuple per line, `t_start, t_end`  \n"
        "Ex:  \n"
        "-4025, -300  \n"
        "-3213, -1232",
        key="e_input",
    )
    col_add.button(
        f"Add to E (cycle {st.session_state.cycle})",
        on_click=add_failures_to_E,
        help=(
            "Merged into the archive at the current cycle."
            if run_started
            else "Before Step 1: kept at cycle 0 and carried into the run Step 1 starts."
        ),
    )
    if "e_message" in st.session_state:
        kind, message = st.session_state.pop("e_message")
        getattr(col_add, kind)(message)

    archive = st.session_state.archive
    failures = archive[archive["verdict"] == 1][["t_start", "t_end", "cycle", "source"]]
    if failures.empty:
        col_list.info("E is empty: mark a failure in the current test set, or add one here.")
    else:
        col_list.write(f"{len(failures)} failure(s) in E (they can't be removed):")
        col_list.dataframe(failures, hide_index=True)


def render_history():
    st.divider()
    st.subheader("Test set development")
    runs = st.session_state.runs
    col_run, col_clear = st.columns([4, 1])
    index = col_run.selectbox(
        "Run", range(len(runs)), index=len(runs) - 1, format_func=lambda i: run_label(i, runs[i])
    )
    if col_clear.button("Clear history", disabled=len(runs) == 1, help="Drop every run but the current one."):
        del runs[:-1]
        st.rerun()
    run = runs[index]
    st.caption(
        "Parameters at Step 1: " + ", ".join(f"`{name}`={value}" for name, value in run["params"].items())
    )

    points = history_chart.run_points(run["archive"])
    max_cycle = int(points["cycle"].max())
    last_cycle = (
        st.slider("Show cycles up to", min_value=1, max_value=max_cycle, value=max_cycle)
        if max_cycle > 1
        else max_cycle
    )
    shown = points[points["cycle"] <= last_cycle]
    metrics = history_chart.cycle_metrics(shown, run["sigma"])

    tested = shown[shown["source"] != "manual"]  # every tuple run on the HiL, overrides too
    m1, m2 = st.columns(2)
    m1.metric("Tests run", len(tested))
    m2.metric("Failures found", int(tested["verdict"].sum()))

    st.altair_chart(
        history_chart.overview_chart(
            shown, *(run["params"][name] for name in ("T_Start", "t_0", "offset", "d_min")), last_cycle
        ),
        width="stretch",
    )
    st.caption(
        "◆ failure (found on the HiL or added to E by hand), ○ pass, coloured by cycle; large = latest cycle shown. "
        "Grey triangle: t_end − t_start < `d_min`, never sampled."
    )

    for col, chart in zip(st.columns(2), history_chart.metrics_charts(metrics)):
        col.altair_chart(chart, width="stretch")


def render_benchmark(errors, blocked_help):
    """The benchmark of WebApp/simulation.py, with the current parameters. Its results stay in
    st.session_state only, and are gone when the browser restarts."""
    st.divider()
    with st.expander("Benchmark (simulated HiL)"):
        st.caption(
            "Runs the EA with the current parameters once per seed, in memory: in a run where no "
            "test fails, to measure how evenly it covers the domain (against as many uniformly "
            "random tests), and against the simulated HiL, to measure how often it finds each "
            "failure region."
        )
        col_seeds, col_cycles, col_run = st.columns(3)
        n_seeds = col_seeds.number_input("Seeds", value=50, min_value=1, max_value=200, step=1, format="%d")
        n_cycles = col_cycles.number_input(
            "Cycles per run", value=60, min_value=1, max_value=500, step=1, format="%d"
        )
        if col_run.button("Run benchmark", disabled=bool(errors), help=blocked_help):
            progress = st.progress(0.0, text=f"Seed 0 of {n_seeds}")
            result = simulation.benchmark(
                range(n_seeds), n_cycles,
                on_progress=lambda done, total: progress.progress(done / total, text=f"Seed {done} of {total}"),
            )
            st.session_state.benchmark = {
                "result": result,
                "params": {name: getattr(ea, name) for name in RUN_PARAMS},
                "seeds": n_seeds,
                "cycles": n_cycles,
            }
            progress.empty()

        if "benchmark" not in st.session_state:
            return
        bench = st.session_state.benchmark
        result = bench["result"]
        st.caption(
            f"{bench['seeds']} seeds (0 to {bench['seeds'] - 1}) × {bench['cycles']} cycles. Parameters: "
            + ", ".join(f"`{name}`={value}" for name, value in bench["params"].items())
        )
        col_gap, col_found = st.columns(2)
        col_gap.altair_chart(history_chart.benchmark_gap_chart(result["coverage_by_cycle"]), width="stretch")
        col_found.altair_chart(history_chart.benchmark_found_chart(result["found_by_cycle"]), width="stretch")

        last = result["coverage_by_cycle"]
        last = last[last["cycle"] == last["cycle"].max()].drop(columns="cycle").set_index("method")
        st.write(f"Coverage after {bench['cycles']} cycles without failures (mean over the seeds):")
        st.dataframe(
            last.style.format({"largest gap (ms)": "{:.0f}", "KS t_start": "{:.3f}", "KS t_end": "{:.3f}"})
        )
        st.caption(
            "Largest gap: the largest distance from any point of the domain to the nearest tested "
            "tuple (`w_explore` metric). KS: Kolmogorov–Smirnov distance of the tested t_start and "
            "t_end from a uniform spread over the domain. Lower is better for both."
        )
        st.write("Failure regions, simulated HiL:")
        st.dataframe(result["regions"].drop(columns="share").set_index("region"))
        runs = result["runs"]
        first = runs["first failure"]
        st.write(
            f"First failure: median cycle {first.median():.0f}, latest {first.max():.0f}; "
            f"runs without any failure: {int(first.isna().sum())} of {len(runs)}. "
            f"Failures found per run: median {runs['failures found'].median():.0f}. "
            "Termination: " + ", ".join(f"{reason} {count}" for reason, count in runs["termination"].value_counts().items())
            + "."
        )


def main():
    st.set_page_config(layout="wide")
    # Nothing is saved from the app: hide the CSV download / copy controls of the tables. The
    # option is sent to the browser at the start of every script run, so it applies from the
    # next run on (no table is shown before Step 1, which triggers one).
    st.set_option("client.disableDataExport", True)
    defaults = parameter_defaults()
    st.title("Evolutionary Algorithm")
    st.write("_All time variables are given in milliseconds (ms)_")
    init_state()

    st.subheader("Domain")
    col1, col2, col3 = st.columns(3)
    ea.T_Start = col1.number_input("`T_Start`: start of the pre-crash recording time", value=defaults["T_Start"])
    ea.t_0 = col2.number_input("`t_0`: end of the pre-crash recording time (normally 0)", value=defaults["t_0"])
    ea.offset = col3.number_input(
        "`offset`: extension of T: the search domain is [T_Start − offset, t_0 + offset]", value=defaults["offset"],
        min_value=0,
    )

    st.subheader("Fitness weights")
    colA, colB, colC, colD, colE = st.columns(5)
    ea.w = colA.number_input(
        "`w`: weight of t_end vs t_start in V1 and d", value=float(defaults["w"]), min_value=0.0, max_value=1.0
    )
    ea.gamma = colB.number_input(
        "`gamma`: weight of V2 (out-of-T penalty)", value=float(defaults["gamma"]), min_value=0.0
    )
    ea.r = colC.number_input(
        "`r`: novelty radius (normalised by |T|)", value=float(defaults["r"]), min_value=0.000001,
        format="%f",
    )
    ea.beta = colD.number_input(
        "`beta`: weight of V3; must be > r", value=float(defaults["beta"]), min_value=0.0, format="%f"
    )
    ea.s = colE.number_input(
        "`s`: ranking selection pressure", value=float(defaults["s"]), min_value=1.0, max_value=2.0
    )

    st.subheader("Mutation")
    colF, colG, colH, colI, colJ = st.columns(5)
    ea.sigma = colF.number_input(
        "`sigma`: initial mutation sigma (ms)", value=float(defaults["sigma"]), min_value=0.0
    )
    ea.patience = colG.number_input(
        "`patience`: cycles without failure before sigma grows", value=defaults["patience"],
        min_value=1, step=1, format="%d",
    )
    ea.immigrant_rate = colH.number_input(
        "`immigrant_rate`: share of random candidates", value=float(defaults["immigrant_rate"]),
        min_value=0.0, max_value=1.0,
    )
    ea.d_min = colI.number_input(
        "`d_min`: minimum event duration (ms)", value=defaults["d_min"], min_value=1, step=1, format="%d"
    )
    ea.resolution_ms = colJ.number_input(
        "`resolution_ms`: boundary resolution (ms)", value=defaults["resolution_ms"], min_value=0,
        step=1, format="%d",
    )
    colP, colQ, colR, _, _ = st.columns(5)
    ea.sigma_min = colP.number_input(
        "`sigma_min`: lower bound of the adaptive sigma (ms)", value=float(defaults["sigma_min"]),
        min_value=0.0,
    )
    ea.sigma_max_ratio = colQ.number_input(
        "`sigma_max_ratio`: upper bound of sigma, as a share of |T|",
        value=float(defaults["sigma_max_ratio"]), min_value=0.000001, format="%f",
    )
    ea.max_attempts = colR.number_input(
        "`max_attempts`: retry rounds to find enough new candidates", value=defaults["max_attempts"],
        min_value=1, step=1, format="%d",
    )

    st.subheader("Population and reproducibility")
    colK, colL, colN, colM = st.columns(4)
    ea.popsize = colK.number_input(
        "`popsize`: test set size", value=defaults["popsize"], min_value=1, step=1, format="%d"
    )
    ea.lambda_new = colL.number_input(
        "`lambda_new`: candidates generated per cycle", value=defaults["lambda_new"], min_value=1,
        step=1, format="%d",
    )
    colL.caption(
        "Must be ≥ `popsize` − `explore_slots`: the test set is picked from these candidates, so "
        "fewer would make it shorter than `popsize`."
    )
    ea.explore_slots = colN.number_input(
        "`explore_slots`: test-set slots for pure exploration", value=defaults["explore_slots"],
        min_value=0, step=1, format="%d",
    )
    ea.explore_pool = colN.number_input(
        "`explore_pool`: Halton points the exploration slots are picked from",
        value=defaults["explore_pool"], min_value=1, step=1, format="%d",
    )
    colN.caption(
        "Must be ≥ `popsize`: while no failure is known every slot explores, taking one tuple of "
        "this pool each, so a smaller pool would make the test set shorter than `popsize`."
    )
    ea.explore_every = colN.number_input(
        "`explore_every`: once a failure is known, every k-th cycle the whole set explores (0 = off)",
        value=defaults["explore_every"], min_value=0, step=1, format="%d",
    )
    ea.w_explore = colN.number_input(
        "`w_explore`: weight of t_end vs t_start in the exploration distance (0.5 = plain)",
        value=float(defaults["w_explore"]), min_value=0.0, max_value=1.0,
    )
    seed_input = colM.text_input("`seed`: RNG seed, applied at Step 1 (blank = random)", value="")

    # Checks across parameters (ea.check_valid_parameters): an error blocks every run button until
    # it is fixed; a caution only warns.
    errors, cautions = ea.check_valid_parameters()
    for message in errors:
        st.error(message)
    for message in cautions:
        st.warning(message)
    blocked_help = "Fix the parameter errors above first." if errors else None

    st.divider()
    st.subheader("HiL loop")
    st.toggle(
        "Simulated HiL: pre-fill “error found” from `simulation.oracle` (more failures can be marked by hand)",
        value=True,
        key="simulated_hil",
    )
    col6, col7, col8 = st.columns(3)

    if col6.button("Step 1: create first test set", disabled=bool(errors), help=blocked_help):
        try:
            seed = int(seed_input) if seed_input.strip() else None
        except ValueError:
            col6.error(f"Invalid seed `{seed_input}`: enter a whole number, or leave it blank.")
        else:
            # A given seed reseeds the run, so the same seed always gives the same first set.
            if seed is not None:
                st.session_state.rng = np.random.default_rng(seed)
            # A new run starts with E = empty, except for the failures added before the first
            # Step 1 (cycle 0), which the first run takes over.
            if st.session_state.test_set.size > 0:
                st.session_state.archive = ea.new_archive()
            st.session_state.sigma = ea.sigma
            st.session_state.cycle = 1
            st.session_state.cycles_without_failure = 0
            st.session_state.no_more_candidates = False
            # A new run; earlier runs stay in the history (see "Test set development").
            st.session_state.runs.append(
                {
                    "seed": seed,
                    "params": {name: getattr(ea, name) for name in RUN_PARAMS},
                    "sigma": {1: ea.sigma},
                }
            )
            # One Halton sequence per run, continued across cycles (as in ea.run_simulation).
            st.session_state.sampler = ea.HaltonSampler(st.session_state.rng)
            record_test_set(
                ea.step_1(rng=st.session_state.rng, sampler=st.session_state.sampler),
                st.session_state.cycle,
            )

    if st.session_state.test_set.size == 0:
        st.info("Run Step 1 to create the first test set.")
    else:
        changed = [
            name for name, value in st.session_state.runs[-1]["params"].items() if getattr(ea, name) != value
        ]
        if changed:
            st.warning(
                f"Parameters changed since Step 1 of this run: {', '.join(changed)}. "
                "The run can no longer be reproduced from its seed."
            )
        st.write(f"Current test set (cycle {st.session_state.cycle}, sigma={st.session_state.sigma:.1f}):")
        editor_key = f"test_set_editor_{st.session_state.test_set_version}"
        st.data_editor(
            test_set_dataframe(st.session_state.test_set, current_verdicts()),
            key=editor_key,
            on_change=apply_editor_edits,
            args=(editor_key,),
            hide_index=True,
            column_config={"error found": st.column_config.CheckboxColumn(required=True)},
            disabled=["t_start", "t_end"],
        )
        st.caption("Checking “error found” adds the tuple to E at once; a failure can't be unchecked.")
        if "table_message" in st.session_state:
            st.warning(st.session_state.pop("table_message"))
        # The only automatic end of a run: Step 2 found no untested candidate.
        finished = st.session_state.no_more_candidates
        if finished:
            st.info("No untested candidates left: the run is finished. Start a new run with Step 1.")
        run_help = blocked_help or ("The run is finished: start a new one with Step 1." if finished else None)

        if col7.button(
            "Propose next test set (Step 2)", disabled=bool(errors) or finished, help=run_help
        ):
            try:
                propose_next_test_set()
            except Exception as e:
                logger.exception("Error proposing the next test set")
                col7.error(f"Error proposing the next test set: {e}")
            else:
                # The table above was already drawn with the old set; redraw with the new one.
                st.rerun()

        # Up to 500 cycles per press: each cycle compares the candidates with the whole archive,
        # so long runs get slower, and the page waits until all of them are done.
        n_cycles = col8.number_input(
            "Cycles to run", value=20, min_value=1, max_value=500, step=1, format="%d"
        )
        if col8.button(
            "Run N cycles (simulated HiL)",
            disabled=not st.session_state.simulated_hil or bool(errors) or finished,
            help=run_help or "Propose and test N more sets in a row, with the oracle's verdicts.",
        ):
            progress = col8.progress(0.0, text=f"Cycle 0 of {n_cycles}")
            try:
                for done in range(n_cycles):
                    if not propose_next_test_set():
                        st.session_state.run_message = (
                            f"Ran {done} of {n_cycles} cycles: no untested candidates left."
                        )
                        break
                    progress.progress((done + 1) / n_cycles, text=f"Cycle {done + 1} of {n_cycles}")
            except Exception as e:
                logger.exception("Error running cycles")
                col8.error(f"Error running cycles: {e}")
            else:
                st.rerun()
        if "run_message" in st.session_state:
            col8.info(st.session_state.pop("run_message"))

    render_error_hit_set()

    if st.session_state.runs:
        # The current run's archive changes on every edit; keep its history entry in sync.
        st.session_state.runs[-1]["archive"] = st.session_state.archive
        render_history()

    render_benchmark(errors, blocked_help)

    st.divider()
    if not st.session_state.archive.empty:
        st.subheader("Archive")
        st.dataframe(st.session_state.archive)


if __name__ == "__main__":
    main()
