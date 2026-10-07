# Reformulate the fitness function (V1, V2, merged V3)

## Context
The user asked whether V1 and V2 in README.md are correct, and whether they can be simplified, merged or written better. They also asked how the novelty term V3 (from EA_concept_review.md) would fit into them. The goal is a fitness function that is correct, normalised and has fewer parameters, so that it follows both defect clustering and the Pesticide Paradox.

## Findings
- **V1 is correct as code, but the scaling is arbitrary.**
  - Dividing by 1000 is a unit change (ms → s), not a normalisation. With a = 1 and b = 3, V1 can reach 20, so the README's "one digit" claim is false.
  - Only the ratio a : b matters, so one of the two parameters is redundant.
  - The notation E_i(t_start) reads like a function call.
- **V1 did not match the intended concept (fixed in README.md and both `V_1` functions).**
  - The intent is to treat t_start and t_End as independent coordinates: each is drawn toward the nearest error start time or error end time on its own, possibly from different tuples of E.
  - The old formula and code took the minimum of the sums (one min over the weighted sum), which forces both coordinates to come from the same tuple E_i.
  - Correct form: the sum of the minima, with independent indices i and j.
- **V2 is wrong in the README**: it says |t_End − T_Start| for every out-of-range case. The code is right only while t_0 = 0.
  - Evolutionary_Algorithm.py:92 uses `t_end` instead of `t_end - t_0`.
  - The WebApp lets the user change t_0, so the code gives wrong results whenever t_0 ≠ 0.
- **The units don't match**: V1 is in seconds × weights, V2 in ms. The effective weight of V2 is therefore c·1000/b, and the rule c > b > a means nothing.
- **V3 belongs inside V1, not V2.** V1 and V3 use the same distance to different sets (E ⊂ A). V2 is a separate concern: it checks that the sample lies within the regulation window.

## Proposed formulation
```
|T|      = t_0 − T_Start                                         w ∈ (0.5, 1)   (replaces a, b, V_1_scale)
E^s      = { e^s : e ∈ E }        all start times that produced an error
E^e      = { e^e : e ∈ E }        all end times that produced an error

D_E(x)   = [ (1−w)·min_i |x_s − e_i^s| + w·min_j |x_e − e_j^e| ] / |T|      i, j independent   (0 if E = ∅)
d(x, y)  = [ (1−w)·|x_s − y_s| + w·|x_e − y_e| ] / |T|                       distance between two whole tuples
D_A(x)   = min_{y∈A} d(x, y)      (A = all tested samples, E ⊂ A; r if A = ∅)

P(x)     = D_E(x) + β·max(0, 1 − D_A(x)/r)        proximity term = V1 merged with V3
V2(x)    = max(0, T_Start − x_e, x_e − t_0) / |T|
f(x)     = P(x) + γ·V2(x)                                       (replaces c)
```
- D_E treats the coordinates independently. Its zero points are the whole grid E^s × E^e: every combination of a known error start time and a known error end time, not only the tuples of E.
  - Example: with E = {(−4025, −300), (−3213, −1232)}, the cross combinations (−4025, −1232) and (−3213, −300) also give D_E = 0.
  - Assumption to justify in the thesis: the error depends on each trigger time separately, not on their combination.
  - Some cross combinations can have x_s > x_e; the existing swap repair handles them.
- D_A (novelty) stays two-dimensional, measured to whole tested tuples. Per-coordinate novelty would forbid ever reusing a known start time.
- A cross combination that has not been tested yet gives P = 0, which is intended: it is a genuinely new test.
- Around a tested point, the minimum lies on a ring of radius r, provided β > r.
- With E = ∅, P reduces to pure novelty search, which is pure exploration.
- f ≥ 0 still holds, but a repeated point is never 0. Even so, selection should switch to rank or tournament selection.
- Alternative: if test events outside T are of no value, apply a hard constraint (clip or reject t_End) and drop V2 and γ entirely.

## Implementation (once approved)
- Evolutionary_Algorithm.py:
  - Replace `a`, `b` and `V_1_scale` with `w`, and `c` with `gamma`.
  - Add `r`, `beta` and the archive `tested_set`.
  - Rewrite `V_1` as a `proximity` function. D_E keeps the per-coordinate minimum already in `V_1` (min over `axis=1` before the weighted sum). D_A uses the whole-tuple distance (weighted sum before the min).
  - Fix `V_2` so it uses `t_0` and divides by |T|.
- Mirror the changes in WebApp/Evolutionary_Algorithm.py and replace the a/b/c inputs in WebApp/layout.py.
- Rewrite the README "Fitness Orientation" section with the formulas above.

## Verification
- Unit check with E = {(−4025, −300)}, A = E, w = 0.75, r = 0.01, β = 0.05:
  - f(E point) = β.
  - f is lowest at distance ≈ r.
  - f rises again beyond r.
- With E = {(−4025, −300), (−3213, −1232)}, confirm that D_E = 0 for the cross combinations (−4025, −1232) and (−3213, −300).
- Set t_0 = 100 and confirm that V2(t_e = 50) = 0.
- Run `python Evolutionary_Algorithm.py` and `streamlit run WebApp/layout.py`.
