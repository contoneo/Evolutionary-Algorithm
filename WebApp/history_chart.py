"""History chart of one HiL run: every test set the EA proposed, layered over the domain T,
plus per-cycle metrics (hit rate, mutation sigma)."""

import altair as alt
import numpy as np
import pandas as pd

FAIL_COLOR = "#d62728"
DOMAIN_COLOR = "#7f7f7f"


def run_points(archive):
    # One row per archived tuple, with its result: "known error" for the manual error_hit_set
    # at cycle 0, then "fail" / "pass".
    points = pd.DataFrame(
        {
            "t_Start": archive["t_start"].astype(int),
            "t_End": archive["t_End"].astype(int),
            "cycle": archive["cycle"].astype(int),
            "verdict": archive["verdict"].astype(int),
        }
    ).reset_index(drop=True)
    points["result"] = np.where(
        points["cycle"] == 0, "known error", np.where(points["verdict"] == 1, "fail", "pass")
    )
    return points


def cycle_metrics(points, sigma_by_cycle):
    # One row per cycle: hit rate of that cycle's test set, and the sigma it was proposed with.
    rows = []
    for cycle in range(1, int(points["cycle"].max()) + 1):
        tested = points[points["cycle"] == cycle]
        rows.append(
            {
                "cycle": cycle,
                "hit rate": tested["verdict"].mean() if len(tested) else np.nan,
                "sigma": sigma_by_cycle.get(cycle, np.nan),
            }
        )
    return pd.DataFrame(rows)


def overview_chart(points, T_Start, t_0, offset, d_min, last_cycle):
    # The whole search space: T as a dotted box, the shaded triangle below the diagonal where
    # t_End - t_start < d_min, which the EA never samples (see _repair), and every tested tuple.
    lo, hi = T_Start - offset, t_0 + offset
    x_scale = alt.Scale(domain=[lo, hi], nice=False, zero=False)
    y_scale = alt.Scale(domain=[lo, hi], nice=False, zero=False)

    def x_enc(field):
        return alt.X(f"{field}:Q", scale=x_scale, title="t_Start (ms)")

    def y_enc(field):
        return alt.Y(f"{field}:Q", scale=y_scale, title="t_End (ms)")

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

    # Passes are circles coloured by cycle; every failure, known (cycle 0) or found, is a red
    # diamond. The latest cycle shown is drawn larger.
    points = points.assign(
        latest=lambda df: np.where(df["cycle"] == last_cycle, "latest cycle", "earlier cycles")
    )
    latest_scale = {"domain": ["latest cycle", "earlier cycles"]}
    point_tooltip = ["t_Start:Q", "t_End:Q", "cycle:Q", "result:N"]
    passes = points[(points["cycle"] > 0) & (points["verdict"] == 0)]
    layers.append(
        alt.Chart(passes)
        .mark_point(shape="circle", filled=False, strokeWidth=2, clip=True)
        .encode(
            x=x_enc("t_Start"),
            y=y_enc("t_End"),
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
                x=x_enc("t_Start"),
                y=y_enc("t_End"),
                size=alt.Size("latest:N", scale=alt.Scale(range=[200, 110], **latest_scale), legend=None),
                tooltip=point_tooltip,
            )
        )
    return alt.layer(*layers).properties(height=560, title="Search space T").interactive()


def metrics_charts(metrics):
    # Two small line charts over the cycles: hit rate and sigma.
    base = alt.Chart(metrics).encode(x=alt.X("cycle:Q", title="cycle", axis=alt.Axis(tickMinStep=1)))
    hit_rate = base.mark_line(point=True, strokeJoin="round").encode(
        y=alt.Y("hit rate:Q", title="hit rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["cycle:Q", alt.Tooltip("hit rate:Q", format=".0%")],
    ).properties(height=200, title="Failures per test, by cycle")
    sigma = base.mark_line(point=True, strokeJoin="round").encode(
        y=alt.Y("sigma:Q", title="sigma (ms)"),
        tooltip=["cycle:Q", alt.Tooltip("sigma:Q", format=".1f")],
    ).properties(height=200, title="Mutation sigma")
    return hit_rate, sigma
