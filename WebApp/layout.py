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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def test_set_dataframe(test_set, verdicts):
    df = pd.DataFrame(test_set, columns=("t_Start", "t_End"))
    df["error found"] = verdicts
    return df


def reset_verdicts():
    st.session_state.verdicts = np.zeros(len(st.session_state.test_set), dtype=bool)


def record_test_set(test_set, cycle):
    # Record a freshly generated test set into the archive immediately (verdicts default to
    # "no error found"), and remember its archive row labels so later checkbox edits
    # (apply_editor_edits) can be written straight back into those same rows.
    st.session_state.archive = ea.record_results(
        st.session_state.archive, test_set, np.zeros(len(test_set), dtype=int), cycle
    )
    st.session_state.current_rows = st.session_state.archive.tail(len(test_set)).index.tolist()


def apply_editor_edits(editor_key):
    # Mirror the editor's checkbox edits into session_state so the marks survive reruns
    # (they are only cleared by reset_verdicts when a new test set is created), and write the
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
    colK, colL, colM = st.columns(3)
    ea.popsize = colK.number_input("`popsize`: test set size", value=ea.popsize, step=1, format="%d")
    ea.lambda_new = colL.number_input(
        "`lambda_new`: candidates generated per cycle", value=ea.lambda_new, step=1, format="%d"
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
    col6, col7 = st.columns(2)

    if col6.button("Step 1: create first test set"):
        st.session_state.archive = ea.merge_manual_error_hit_set(ea.new_archive(), ea.error_hit_set)
        st.session_state.test_set = ea.initialize_samples_array(ea.popsize, rng=st.session_state.rng)
        st.session_state.test_set_version += 1
        st.session_state.sigma = ea.sigma
        st.session_state.cycle = 1
        st.session_state.cycles_without_failure = 0
        reset_verdicts()
        record_test_set(st.session_state.test_set, st.session_state.cycle)

    if st.session_state.test_set.size == 0:
        st.info("Run Step 1 to create the first test set.")
    else:
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
            has_failure = bool(st.session_state.verdicts.sum() > 0)
            new_sigma, new_cycles_without_failure = ea.update_sigma(
                st.session_state.sigma, has_failure, st.session_state.cycles_without_failure
            )
            try:
                next_test_set = ea.step_2(
                    st.session_state.archive,
                    st.session_state.test_set,
                    new_sigma,
                    rng=st.session_state.rng,
                )
            except Exception as e:
                logger.exception("Error proposing the next test set")
                col7.error(f"Error proposing the next test set: {e}")
            else:
                st.session_state.sigma = new_sigma
                st.session_state.cycles_without_failure = new_cycles_without_failure
                st.session_state.cycle += 1
                st.session_state.test_set = next_test_set
                st.session_state.test_set_version += 1
                reset_verdicts()
                record_test_set(next_test_set, st.session_state.cycle)
                # The table above was already drawn with the old set; redraw with the new one.
                st.rerun()

    st.divider()
    if not st.session_state.archive.empty:
        st.subheader("Archive")
        st.dataframe(st.session_state.archive)


if __name__ == "__main__":
    main()
