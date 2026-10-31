"""History chart of one HiL run: every test set the EA proposed, layered over the domain T,
plus per-cycle metrics (failures per test, mutation sigma)."""

import altair as alt
import numpy as np
import pandas as pd

FAIL_COLOR = "#d62728"
DOMAIN_COLOR = "#7f7f7f"


def run_points(archive):
    """One row per archived tuple, with its result: "manual failure" for a failure added to E by
    hand, "override" for an earlier pass marked as failed by hand, otherwise the HiL verdict,
    "fail" / "pass"."""
    points = pd.DataFrame(
        {
            "t_start": archive["t_start"].astype(int),
            "t_end": archive["t_end"].astype(int),
            "cycle": archive["cycle"].astype(int),
            "verdict": archive["verdict"].astype(int),
            "source": archive["source"].astype(str),
        }
    ).reset_index(drop=True)
    points["result"] = np.select(
        [points["source"] == "manual", points["source"] == "override", points["verdict"] == 1],
        ["manual failure", "override", "fail"],
        default="pass",
    )
    return points


def cycle_metrics(points, sigma_by_cycle):
    """One row per cycle: failures per test of that cycle's test set (HiL tests only, not the failures
    added by hand), and the sigma it was proposed with."""
    rows = []
    for cycle in range(1, int(points["cycle"].max()) + 1):
        tested = points[(points["cycle"] == cycle) & (points["source"] == "hil")]
        rows.append(
            {
                "cycle": cycle,
                "failures per test": tested["verdict"].mean() if len(tested) else np.nan,
                "sigma": sigma_by_cycle.get(cycle, np.nan),
            }
        )
    return pd.DataFrame(rows)


def overview_chart(points, T_Start, t_0, offset, d_min, last_cycle):
    """The whole search space: T as a dotted box, the shaded triangle below the diagonal where
    t_end - t_start < d_min, which the EA never samples (see _repair), and every tested tuple."""
    lo, hi = T_Start - offset, t_0 + offset
    x_scale = alt.Scale(domain=[lo, hi], nice=False, zero=False)
    y_scale = alt.Scale(domain=[lo, hi], nice=False, zero=False)

    def x_enc(field):
        return alt.X(f"{field}:Q", scale=x_scale, title="t_start (ms)")

    def y_enc(field):
        return alt.Y(f"{field}:Q", scale=y_scale, title="t_end (ms)")

    infeasible = pd.DataFrame({"x": [lo, hi], "y": [lo + d_min, hi + d_min], "y2": [lo, lo]})
    window = pd.DataFrame([{"x": T_Start, "x2": t_0, "y": T_Start, "y2": t_0}])
    layers = [
        alt.Chart(infeasible)
        .mark_area(color=DOMAIN_COLOR, opacity=0.12, clip=True)
        .encode(x=x_enc("x"), y=y_enc("y"), y2="y2"),
        alt.Chart(window)
        .mark_rect(fillOpacity=0, stroke=DOMAIN_COLOR, strokeDash=[2, 2], clip=True)
        .encode(x=x_enc("x"), x2="x2", y=y_enc("y"), y2="y2"),
    ]

    # Passes are circles coloured by cycle; every failure, found on the HiL or added by hand, is
    # a red diamond. The latest cycle shown is drawn larger.
    points = points.assign(
        latest=lambda df: np.where(df["cycle"] == last_cycle, "latest cycle", "earlier cycles")
    )
    latest_scale = {"domain": ["latest cycle", "earlier cycles"]}
    point_tooltip = ["t_start:Q", "t_end:Q", "cycle:Q", "result:N"]
    passes = points[points["verdict"] == 0]
    layers.append(
        alt.Chart(passes)
        .mark_point(shape="circle", filled=False, strokeWidth=2, clip=True)
        .encode(
            x=x_enc("t_start"),
            y=y_enc("t_end"),
            color=alt.Color("cycle:Q", scale=alt.Scale(scheme="viridis", reverse=True), title="cycle"),
            size=alt.Size("latest:N", scale=alt.Scale(range=[140, 45], **latest_scale), title=None),
            opacity=alt.Opacity("latest:N", scale=alt.Scale(range=[1.0, 0.55], **latest_scale), legend=None),
            tooltip=point_tooltip,
        )
    )
    failures = points[points["verdict"] == 1]
    if not failures.empty:
        layers.append(
            alt.Chart(failures)
            .mark_point(shape="diamond", filled=True, color=FAIL_COLOR, clip=True)
            .encode(
                x=x_enc("t_start"),
                y=y_enc("t_end"),
                size=alt.Size("latest:N", scale=alt.Scale(range=[200, 110], **latest_scale), legend=None),
                tooltip=point_tooltip,
            )
        )
    return alt.layer(*layers).properties(height=560, title=f"Search space [{lo}, {hi}] (dotted box: T)").interactive()


def metrics_charts(metrics):
    """Two small line charts over the cycles: failures per test and sigma."""
    base = alt.Chart(metrics).encode(x=alt.X("cycle:Q", title="cycle", axis=alt.Axis(tickMinStep=1)))
    hit_rate = base.mark_line(point=True, strokeJoin="round").encode(
        y=alt.Y("failures per test:Q", title="failures per test", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["cycle:Q", alt.Tooltip("failures per test:Q", format=".0%")],
    ).properties(height=200, title="Failures per test, by cycle")
    sigma = base.mark_line(point=True, strokeJoin="round").encode(
        y=alt.Y("sigma:Q", title="sigma (ms)"),
        tooltip=["cycle:Q", alt.Tooltip("sigma:Q", format=".1f")],
    ).properties(height=200, title="Mutation sigma")
    return hit_rate, sigma


# Two-series colours of the benchmark charts (categorical slots 1 and 2), validated for colour-
# vision deficiencies and contrast on both the light and the dark surface.
SERIES_COLORS = ["#2a78d6", "#d95926"]


def benchmark_gap_chart(coverage_by_cycle):
    """Largest untested gap by cycle, mean over the seeds: the EA (no failures, coverage mode)
    against as many uniformly random tests."""
    return (
        alt.Chart(coverage_by_cycle)
        .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2, strokeJoin="round")
        .encode(
            x=alt.X("cycle:Q", title="cycle", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("largest gap (ms):Q", title="largest untested gap (ms)"),
            color=alt.Color(
                "method:N", title=None,
                scale=alt.Scale(domain=["EA", "random"], range=SERIES_COLORS),
            ),
            tooltip=["method:N", "cycle:Q", alt.Tooltip("largest gap (ms):Q", format=".0f")],
        )
        .properties(height=240, title="Largest untested gap, no failures (lower is better)")
    )


def benchmark_found_chart(found_by_cycle):
    """Share of runs that have found each failure region of the simulated HiL, by cycle."""
    regions = list(dict.fromkeys(found_by_cycle["region"]))
    return (
        alt.Chart(found_by_cycle)
        .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2, strokeJoin="round")
        .encode(
            x=alt.X("cycle:Q", title="cycle", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y(
                "share found:Q", title="runs that found it",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "region:N", title=None,
                scale=alt.Scale(domain=regions, range=SERIES_COLORS[: len(regions)]),
            ),
            tooltip=["region:N", "cycle:Q", alt.Tooltip("share found:Q", format=".0%")],
        )
        .properties(height=240, title="Failure regions found, simulated HiL")
    )
