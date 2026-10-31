import os

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

LAYOUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "WebApp", "layout.py")


def _button(at, label):
    return next(b for b in at.button if b.label.startswith(label))


def _checkbox_edits_script():
    """Runs inside a Streamlit script: check a pass in the current test set, then try to
    uncheck it again (the editor's on_change callback, fed with the editor's edited_rows)."""
    import numpy as np
    import streamlit as st

    from WebApp import layout

    layout.init_state()
    if "result" not in st.session_state:
        st.session_state.simulated_hil = False  # every tuple starts as a pass
        layout.record_test_set(np.array([[-2000, -500], [-1000, -100]]), 1)
        st.session_state["editor"] = {"edited_rows": {0: {"error found": True}}}
        layout.apply_editor_edits("editor")
        checked = layout.current_verdicts().tolist()
        version = st.session_state.test_set_version
        st.session_state["editor"] = {"edited_rows": {0: {"error found": False}}}
        layout.apply_editor_edits("editor")
        st.session_state.result = {
            "checked": checked,
            "after_uncheck": layout.current_verdicts().tolist(),
            "editor_redrawn": st.session_state.test_set_version > version,
            "message": st.session_state.get("table_message"),
        }


def test_a_checked_failure_joins_E_and_cannot_be_unchecked():
    at = AppTest.from_function(_checkbox_edits_script, default_timeout=30).run()
    result = at.session_state["result"]
    assert result["checked"] == [True, False]
    assert result["after_uncheck"] == [True, False]
    assert result["editor_redrawn"]
    assert "can't be removed" in result["message"]


def test_add_to_E_before_step_1_goes_to_cycle_0_and_afterwards_to_the_current_cycle():
    at = AppTest.from_file(LAYOUT, default_timeout=30).run()
    assert not _button(at, "Add to E").disabled

    # Before Step 1: kept at cycle 0 and carried into the run Step 1 starts.
    at.text_area(key="e_input").input("-4025, -300").run()
    _button(at, "Add to E").click().run()
    at.text_input[0].input("42").run()
    _button(at, "Step 1").click().run()
    archive = at.session_state["archive"]
    assert at.session_state["cycle"] == 1
    manual = archive[archive["source"] == "manual"]
    assert manual[["t_start", "t_end", "cycle"]].values.tolist() == [[-4025, -300, 0]]

    # During the run: merged at the current cycle; the input is cleared.
    at.text_area(key="e_input").input("-3213, -1232").run()
    _button(at, "Add to E").click().run()
    archive = at.session_state["archive"]
    manual = archive[archive["source"] == "manual"]
    assert manual[["t_start", "t_end", "cycle"]].values.tolist() == [[-4025, -300, 0], [-3213, -1232, 1]]
    assert (manual["verdict"] == 1).all()
    assert at.text_area(key="e_input").value == ""

    _button(at, "Propose next test set").click().run()
    _button(at, "Run N cycles").click().run()
    archive = at.session_state["archive"]
    assert not at.exception
    assert (archive["source"] == "manual").sum() == 2  # nothing added or removed them since
    assert archive.duplicated(subset=["t_start", "t_end"]).sum() == 0

    # A later Step 1 starts a new run with E = empty; the previous run stays in the history.
    _button(at, "Step 1").click().run()
    archive = at.session_state["archive"]
    assert (archive["source"] == "hil").all() and (archive["cycle"] == 1).all()
    assert len(at.session_state["runs"]) == 2


def test_a_new_browser_session_starts_fresh():
    """Nothing carries over between browser sessions: parameters, seed and run start anew."""
    first = AppTest.from_file(LAYOUT, default_timeout=30).run()
    popsize = next(n for n in first.number_input if n.label.startswith("`popsize`"))
    default_popsize = popsize.value
    popsize.set_value(default_popsize + 3).run()
    first.text_input[0].input("5").run()
    _button(first, "Step 1").click().run()
    assert len(first.session_state["archive"]) == default_popsize + 3

    second = AppTest.from_file(LAYOUT, default_timeout=30).run()
    assert next(n for n in second.number_input if n.label.startswith("`popsize`")).value == default_popsize
    assert second.text_input[0].value == ""
    assert second.session_state["archive"].empty
    assert second.session_state["runs"] == []


def _number_input(at, name):
    return next(n for n in at.number_input if n.label.startswith(f"`{name}`"))


def test_parameter_errors_block_the_run_and_cautions_only_warn():
    at = AppTest.from_file(LAYOUT, default_timeout=30).run()
    assert not at.error and not at.warning
    _number_input(at, "popsize").set_value(30).run()
    assert any("lambda_new" in e.value for e in at.error)
    assert _button(at, "Step 1").disabled
    _number_input(at, "popsize").set_value(4).run()
    assert not at.error and not _button(at, "Step 1").disabled

    _number_input(at, "explore_slots").set_value(3).run()
    _number_input(at, "explore_pool").set_value(2).run()
    assert any("explore_pool" in e.value for e in at.error)
    assert _button(at, "Step 1").disabled
    _number_input(at, "explore_pool").set_value(20).run()
    _number_input(at, "explore_slots").set_value(1).run()
    assert not at.error

    _number_input(at, "beta").set_value(0.005).run()
    assert any("beta" in w.value for w in at.warning)
    assert not _button(at, "Step 1").disabled


def _cycles_to_run(at):
    return next(n for n in at.number_input if n.label == "Cycles to run")


def test_run_n_cycles_runs_exactly_n_cycles_per_press():
    at = AppTest.from_file(LAYOUT, default_timeout=30).run()
    _button(at, "Step 1").click().run()
    _cycles_to_run(at).set_value(3).run()
    _button(at, "Run N cycles").click().run()
    assert at.session_state["cycle"] == 1 + 3
    _button(at, "Run N cycles").click().run()
    assert at.session_state["cycle"] == 1 + 3 + 3
    assert not at.exception


@pytest.mark.filterwarnings("ignore:Only .* candidates available")  # expected near the end
def test_no_more_candidates_finishes_the_run_until_step_1():
    at = AppTest.from_file(LAYOUT, default_timeout=60).run()
    _number_input(at, "T_Start").set_value(-6).run()  # [-6, 0]: 21 possible tuples
    _number_input(at, "offset").set_value(0).run()
    _number_input(at, "sigma_max_ratio").set_value(1.0).run()  # sigma up to 6 ms, above sigma_min
    at.text_input[0].input("0").run()
    _button(at, "Step 1").click().run()
    _cycles_to_run(at).set_value(50).run()
    _button(at, "Run N cycles").click().run()

    assert any("No untested candidates left" in i.value for i in at.info)
    assert any("Ran" in i.value and "of 50 cycles" in i.value for i in at.info)
    assert len(at.session_state["archive"]) == 21
    assert _button(at, "Propose next test set").disabled
    assert _button(at, "Run N cycles").disabled

    _button(at, "Step 1").click().run()  # a new run: the buttons work again
    assert not any("No untested candidates left" in i.value for i in at.info)
    assert not _button(at, "Propose next test set").disabled
    assert not _button(at, "Run N cycles").disabled


def test_simulation_is_part_of_the_web_app_only():
    from WebApp import simulation

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert not os.path.exists(os.path.join(repo_root, "simulation.py"))
    assert not hasattr(simulation, "main")


def test_sigma_bounds_and_max_attempts_are_set_in_the_app_and_saved_with_the_run():
    at = AppTest.from_file(LAYOUT, default_timeout=30).run()
    _number_input(at, "sigma_min").set_value(2.0).run()
    _number_input(at, "max_attempts").set_value(7).run()
    _button(at, "Step 1").click().run()
    params = at.session_state["runs"][-1]["params"]
    assert params["sigma_min"] == 2.0 and params["max_attempts"] == 7
    assert "sigma_max_ratio" in params

    _number_input(at, "sigma_min").set_value(600.0).run()  # above 0.1 * 5000 = 500 ms
    assert any("sigma_min" in e.value for e in at.error)
    assert _button(at, "Propose next test set").disabled


def test_seeded_web_app_run_equals_run_simulation():
    import Evolutionary_Algorithm as ea
    from WebApp import simulation

    for known_failure in ("", "-4025, -300"):
        at = AppTest.from_file(LAYOUT, default_timeout=60).run()
        if known_failure:  # added before Step 1: cycle 0, like error_hit_set in run_simulation
            at.text_area(key="e_input").input(known_failure).run()
            _button(at, "Add to E").click().run()
        at.text_input[0].input("42").run()
        _button(at, "Step 1").click().run()
        _cycles_to_run(at).set_value(10).run()
        _button(at, "Run N cycles").click().run()
        web = at.session_state["archive"].reset_index(drop=True)

        ea.error_hit_set = ea.parse_error_hit_set(known_failure)  # reset after each test
        sim, _, _ = ea.run_simulation(simulation.oracle, cycles=11, seed=42)
        assert web.equals(sim.reset_index(drop=True))


def test_benchmark_section_shows_its_results_and_is_blocked_by_parameter_errors():
    at = AppTest.from_file(LAYOUT, default_timeout=60).run()
    next(n for n in at.number_input if n.label == "Seeds").set_value(2).run()
    next(n for n in at.number_input if n.label == "Cycles per run").set_value(10).run()
    _button(at, "Run benchmark").click().run()
    assert not at.exception
    tables = [table.value.index.tolist() for table in at.dataframe]
    assert ["EA", "random"] in tables and ["region_1", "region_2", "all regions"] in tables
    assert any("First failure" in text.value for text in at.markdown)

    _number_input(at, "explore_pool").set_value(2).run()  # below popsize: a parameter error
    assert _button(at, "Run benchmark").disabled
