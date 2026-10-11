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
import simulation
from WebApp import history_chart

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Parameters saved with each run at Step 1, so a run can be traced back to its settings.
RUN_PARAMS = (
    "T_Start", "t_0", "offset", "w", "gamma", "r", "beta", "s", "sigma", "patience",
    "immigrant_rate", "d_min", "resolution_ms", "popsize", "lambda_new", "explore_slots",
)


def parse_input(input_text):
    rows = input_text.strip().split("\n")
    if rows == [""]:
        return np.array([])
    array_data = [list(map(int, row.split(","))) for row in rows]
    return np.unique(np.array(array_data), axis=0)


def init_state():
    # Per-session state: the archive, current test set, sigma and cycle are stateful and
    # belong to this HiL session, unlike the module-level parameters below (w, gamma, popsize,
    # ...), which stay plain `ea.<param>` globals shared by every session, same as before.
    st.session_state.setdefault("archive", ea.new_archive())
    st.session_state.setdefault("test_set", np.array([]))
    st.session_state.setdefault("test_set_version", 0)
    st.session_state.setdefault("verdicts", np.array([], dtype=bool))
    st.session_state.setdefault("current_rows", [])
    st.session_state.setdefault("sigma", ea.sigma)
    st.session_state.setdefault("cycle", 0)
    st.session_state.setdefault("cycles_without_failure", 0)
    st.session_state.setdefault("rng", ea.rng)
    # Every run started by Step 1 (seed, parameters, archive), kept until "Clear history".
    st.session_state.setdefault("runs", [])


def test_set_dataframe(test_set, verdicts):
    df = pd.DataFrame(test_set, columns=("t_Start", "t_End"))
    df["error found"] = verdicts
    return df


def initial_verdicts(test_set):
    # With the simulated HiL on, the oracle pre-fills the verdicts (still editable as a manual
    # override); otherwise they default to "no error found".
    if st.session_state.get("simulated_hil", True):
        return np.array([bool(simulation.oracle(tuple(row))) for row in test_set], dtype=bool)
    return np.zeros(len(test_set), dtype=bool)


def record_test_set(test_set, cycle):
    # Make test_set the current one and record it into the archive immediately, with its
    # initial verdicts, and remember its archive row labels so later checkbox edits
    # (apply_editor_edits) can be written straight back into those same rows.
    st.session_state.test_set = test_set
    st.session_state.test_set_version += 1
    st.session_state.verdicts = initial_verdicts(test_set)
    st.session_state.archive = ea.record_results(
        st.session_state.archive, test_set, st.session_state.verdicts.astype(int), cycle
    )
    st.session_state.current_rows = st.session_state.archive.tail(len(test_set)).index.tolist()


def propose_next_test_set():
    # Step 2, in the same order as ea.run_simulation: adapt sigma from the current set's
    # verdicts, propose the next set, record it. State changes only once step_2 succeeded.
    has_failure = bool(st.session_state.verdicts.sum() > 0)
    new_sigma, new_cycles_without_failure = ea.update_sigma(
        st.session_state.sigma, has_failure, st.session_state.cycles_without_failure
    )
    next_test_set = ea.step_2(
        st.session_state.archive, st.session_state.test_set, new_sigma, rng=st.session_state.rng
    )
    if next_test_set.shape[0] == 0:
        raise ValueError("no untested candidates left")
    st.session_state.sigma = new_sigma
    st.session_state.cycles_without_failure = new_cycles_without_failure
    st.session_state.cycle += 1
    record_test_set(next_test_set, st.session_state.cycle)
    st.session_state.runs[-1]["sigma"][st.session_state.cycle] = new_sigma


def boundary_resolved():
    archive = st.session_state.archive
    return ea.boundary_resolved(ea.archive_error_set(archive), ea.archive_passed_set(archive))


def run_label(index, run):
    seed = "random" if run["seed"] is None else run["seed"]
    cycles = int(run["archive"]["cycle"].max())
    return f"Run {index + 1}: seed {seed}, {run['known_errors']} known error(s), {cycles} cycle(s)"


def apply_editor_edits(editor_key):
    # Mirror the editor's checkbox edits into session_state so the marks survive reruns
    # (they are only replaced when record_test_set makes a new test set current), and write the
    # same edit straight into the archive row already recorded for this test set.
    edits = st.session_state[editor_key].get("edited_rows", {})
    verdicts = st.session_state.verdicts.copy()
    archive = st.session_state.archive
    for row, changes in edits.items():
        if "error found" in changes:
            value = bool(changes["error found"])
            verdicts[int(row)] = value
            archive.loc[st.session_state.current_rows[int(row)], "verdict"] = int(value)
    st.session_state.verdicts = verdicts
    st.session_state.archive = archive


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

    tested = shown[shown["cycle"] > 0]
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
        "◆ failure (known from `error_hit_set` or found by the EA), ○ pass, coloured by cycle; large = latest cycle shown. "
        "Grey triangle: t_End − t_Start < `d_min`, never sampled."
    )

    for col, chart in zip(st.columns(2), history_chart.metrics_charts(metrics)):
        col.altair_chart(chart, width="stretch")


def main():
    st.set_page_config(layout="wide")
    st.title("Evolutionary Algorithm")
    st.write("_All time variables are given in milliseconds (ms)_")
    init_state()

    st.subheader("Domain")
    col1, col2, col3 = st.columns(3)
    ea.T_Start = col1.number_input("`T_Start`: start of the pre-crash recording time", value=ea.T_Start)
    ea.t_0 = col2.number_input("`t_0`: end of the pre-crash recording time (normally 0)", value=ea.t_0)
    ea.offset = col3.number_input(
        "`offset`: extension of T for Step 1 / random immigrant samples", value=ea.offset
    )

    st.subheader("Fitness weights")
    colA, colB, colC, colD, colE = st.columns(5)
    ea.w = colA.number_input(
        "`w`: weight of t_End vs t_start in V1 and d", value=float(ea.w), min_value=0.0, max_value=1.0
    )
    ea.gamma = colB.number_input("`gamma`: weight of V2 (out-of-T penalty)", value=float(ea.gamma))
    ea.r = colC.number_input("`r`: novelty radius (normalised by |T|)", value=float(ea.r), format="%f")
    ea.beta = colD.number_input("`beta`: weight of V3; must be > r", value=float(ea.beta), format="%f")
    ea.s = colE.number_input(
        "`s`: ranking selection pressure", value=float(ea.s), min_value=1.0, max_value=2.0
    )

    st.subheader("Mutation")
    colF, colG, colH, colI, colJ = st.columns(5)
    ea.sigma = colF.number_input("`sigma`: initial mutation sigma (ms)", value=float(ea.sigma))
    ea.patience = colG.number_input(
        "`patience`: cycles without failure before sigma grows", value=ea.patience, step=1, format="%d"
    )
    ea.immigrant_rate = colH.number_input(
        "`immigrant_rate`: share of random candidates", value=float(ea.immigrant_rate),
        min_value=0.0, max_value=1.0,
    )
    ea.d_min = colI.number_input("`d_min`: minimum event duration (ms)", value=ea.d_min, step=1, format="%d")
    ea.resolution_ms = colJ.number_input(
        "`resolution_ms`: boundary resolution (ms)", value=ea.resolution_ms, step=1, format="%d"
    )

    st.subheader("Population and reproducibility")
    colK, colL, colN, colM = st.columns(4)
    ea.popsize = colK.number_input("`popsize`: test set size", value=ea.popsize, step=1, format="%d")
    ea.lambda_new = colL.number_input(
        "`lambda_new`: candidates generated per cycle", value=ea.lambda_new, step=1, format="%d"
    )
    ea.explore_slots = colN.number_input(
        "`explore_slots`: test-set slots for pure exploration", value=ea.explore_slots,
        min_value=0, step=1, format="%d",
    )
    seed_input = colM.text_input("`seed`: RNG seed (blank = random)", value="" if ea.seed is None else str(ea.seed))
    if colM.button("Apply seed"):
        new_seed = int(seed_input) if seed_input.strip() else None
        st.session_state.rng = ea.seed_rng(new_seed)
        st.success(f"RNG reseeded (seed={new_seed}).")

    st.divider()
    st.subheader("Error hit set and archive")
    col4, col5 = st.columns(2)
    input_text = col4.text_area(
        # Markdown collapses single newlines; two trailing spaces force a line break.
        "`error_hit_set`: one tuple per line, `t_Start, t_End`  \n"
        "Ex:  \n"
        "-4025, -300  \n"
        "-3213, -1232"
    )
    if col4.button("Enter `error_hit_set`"):
        try:
            ea.error_hit_set = parse_input(input_text)
            col4.success("`error_hit_set` input successful!")
        except ValueError as e:
            logger.exception("Invalid error_hit_set input")
            col4.error(f"Invalid input format. Please enter a valid `error_hit_set`. ({e})")

    if not st.session_state.archive.empty:
        col5.download_button(
            "Download archive CSV",
            st.session_state.archive.to_csv(index=False),
            file_name="archive.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("HiL loop")
    st.toggle(
        "Simulated HiL: pre-fill “error found” from `simulation.oracle` (marks stay editable)",
        value=True,
        key="simulated_hil",
    )
    col6, col7, col8 = st.columns(3)

    if col6.button("Step 1: create first test set"):
        try:
            seed = int(seed_input) if seed_input.strip() else None
        except ValueError:
            col6.error(f"Invalid seed `{seed_input}`: enter a whole number, or leave it blank.")
        else:
            # A given seed reseeds the run, so the same seed always gives the same first set.
            if seed is not None:
                st.session_state.rng = np.random.default_rng(seed)
            st.session_state.archive = ea.merge_manual_error_hit_set(ea.new_archive(), ea.error_hit_set)
            st.session_state.sigma = ea.sigma
            st.session_state.cycle = 1
            st.session_state.cycles_without_failure = 0
            # A new run; earlier runs stay in the history (see "Test set development").
            st.session_state.runs.append(
                {
                    "seed": seed,
                    "known_errors": int(np.asarray(ea.error_hit_set).reshape(-1, 2).shape[0]),
                    "params": {name: getattr(ea, name) for name in RUN_PARAMS},
                    "sigma": {1: ea.sigma},
                }
            )
            record_test_set(
                ea.initialize_samples_array(ea.popsize, rng=st.session_state.rng), st.session_state.cycle
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
        edited = st.data_editor(
            test_set_dataframe(st.session_state.test_set, st.session_state.verdicts),
            key=editor_key,
            on_change=apply_editor_edits,
            args=(editor_key,),
            hide_index=True,
            column_config={"error found": st.column_config.CheckboxColumn(required=True)},
            disabled=["t_Start", "t_End"],
        )
        st.caption("Marks in “error found” are saved to the archive automatically.")

        if col7.button("Propose next test set (Step 2)"):
            try:
                propose_next_test_set()
            except Exception as e:
                logger.exception("Error proposing the next test set")
                col7.error(f"Error proposing the next test set: {e}")
            else:
                # The table above was already drawn with the old set; redraw with the new one.
                st.rerun()

        n_cycles = col8.number_input("Cycles to run", value=20, min_value=1, step=1, format="%d")
        if col8.button(
            "Run N cycles (simulated HiL)",
            disabled=not st.session_state.simulated_hil,
            help="Propose and test N more sets in a row, with the oracle's verdicts.",
        ):
            try:
                for _ in range(n_cycles):
                    if boundary_resolved():
                        st.session_state.run_message = "Stopped early: failure boundary resolved."
                        break
                    propose_next_test_set()
            except Exception as e:
                logger.exception("Error running cycles")
                col8.error(f"Error running cycles: {e}")
            else:
                st.rerun()
        if "run_message" in st.session_state:
            col8.info(st.session_state.pop("run_message"))

    if st.session_state.runs:
        # The current run's archive changes on every edit; keep its history entry in sync.
        st.session_state.runs[-1]["archive"] = st.session_state.archive
        render_history()

    st.divider()
    if not st.session_state.archive.empty:
        st.subheader("Archive")
        st.dataframe(st.session_state.archive)


if __name__ == "__main__":
    main()
