# Reformulate the fitness function (V1, V2, merged V3)

## Context
The user asked whether V1 and V2 in README.md are correct, and whether they can be simplified, merged or written better. They also asked how the novelty term V3 (from EA_concept_review.md) would fit into them. The goal is a fitness function that is correct, normalised and has fewer parameters, so that it follows both defect clustering and the Pesticide Paradox.

## Findings
- **V1 is correct as code, but the scaling is arbitrary.**
  - Dividing by 1000 is a unit change (ms → s), not a normalisation. With a = 1 and b = 3, V1 can reach 20, so the README's "one digit" claim is false.
  - Only the ratio a : b matters, so one of the two parameters is redundant.
  - The notation E_i(t_start) reads like a function call.
- **V2 is wrong in the README**: it says |t_End − T_Start| for every out-of-range case. The code is right only while t_0 = 0.
  - Evolutionary_Algorithm.py:92 uses `t_end` instead of `t_end - t_0`.
  - The WebApp lets the user change t_0, so the code gives wrong results whenever t_0 ≠ 0.
- **The units don't match**: V1 is in seconds × weights, V2 in ms. The effective weight of V2 is therefore c·1000/b, and the rule c > b > a means nothing.
- **V3 belongs inside V1, not V2.** V1 and V3 use the same distance to different sets (E ⊂ A). V2 is a separate concern: it checks that the sample lies within the regulation window.

## Proposed formulation
```
|T|      = t_0 − T_Start
d(x, y)  = [ (1−w)·|x_s − y_s| + w·|x_e − y_e| ] / |T|        w ∈ (0.5, 1)   (replaces a, b, V_1_scale)
D_E(x)   = min_{e∈E} d(x, e)      (0 if E = ∅)
D_A(x)   = min_{y∈A} d(x, y)      (A = all tested samples, E ⊂ A; r if A = ∅)

P(x)     = D_E(x) + β·max(0, 1 − D_A(x)/r)        proximity term = V1 merged with V3
V2(x)    = max(0, T_Start − x_e, x_e − t_0) / |T|
f(x)     = P(x) + γ·V2(x)                                       (replaces c)
```
- The minimum lies on a ring of radius r around the nearest tested point, provided β > r.
- With E = ∅, P reduces to pure novelty search, which is pure exploration.
- f ≥ 0 still holds, but a repeated point is never 0. Even so, selection should switch to rank or tournament selection.
- Alternative: if test events outside T are of no value, apply a hard constraint (clip or reject t_End) and drop V2 and γ entirely.

## Implementation (once approved)
- Evolutionary_Algorithm.py:
  - Replace `a`, `b` and `V_1_scale` with `w`, and `c` with `gamma`.
  - Add `r`, `beta` and the archive `tested_set`.
  - Rewrite `V_1` as a `proximity` function using one shared helper, `weighted_distance(samples, ref_set)`, called once for E and once for A.
  - Fix `V_2` so it uses `t_0` and divides by |T|.
- Mirror the changes in WebApp/Evolutionary_Algorithm.py and replace the a/b/c inputs in WebApp/layout.py.
- Rewrite the README "Fitness Orientation" section with the formulas above.

## Verification
- Unit check with E = {(−4025, −300)}, A = E, w = 0.75, r = 0.01, β = 0.05:
  - f(E point) = β.
  - f is lowest at distance ≈ r.
  - f rises again beyond r.
- Set t_0 = 100 and confirm that V2(t_e = 50) = 0.
- Run `python Evolutionary_Algorithm.py` and `streamlit run WebApp/layout.py`.
