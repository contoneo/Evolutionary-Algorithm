# Pending issues

State on 2026-10-31: `pytest` 76/76 pass. All changes from the review are uncommitted, on top of commit `2c37c44` ("clean project"). One issue is left.

---

## N14. The benchmark charts haven't been looked at in a browser

The two benchmark charts (largest untested gap by cycle, EA vs random; regions found by cycle) were built and pass the headless web-app test, and their colours were validated for colour-vision deficiencies and contrast on the light and dark surface. They haven't been rendered and looked at in a real browser (label collisions, legend placement, axis ranges). Check: restart the server (`streamlit run WebApp/layout.py`), open **Benchmark (simulated HiL)**, press **Run benchmark**, and look at both charts in light and dark mode.
