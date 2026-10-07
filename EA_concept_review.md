# EA concept review: observations and proposals

## Context
The user asked for a review of the concept in README.md, checked against Evolutionary_Algorithm.py, WebApp/Evolutionary_Algorithm.py and WebApp/layout.py. This is an analysis. Nothing gets changed until the user chooses which proposals to carry out.

## A. Concept-level observations (most important first)

1. **The fitness pulls samples onto known errors, which works against the Pesticide Paradox.**
   - Minimising V1 drives samples toward E, and the optimum is a point equal to a known failure, which is a re-test.
   - That is pure exploitation. "Defect clustering" justifies searching *near* E, but the Pesticide Paradox calls for *new* points.
   - Proposal: score a target distance band (e.g. |d − d*| with d* > 0). Alternatively, add a novelty term that penalises closeness to every sample already tested (an archive), not just to E.

2. **The HiL test results are never fed back into the loop.**
   - The README says tests run between generations, but `EA()` runs `step_2_iterations` generations on the surrogate fitness alone. `error_hit_set` is never updated.
   - Passing samples carry information too (where the error is *not*), but they are thrown away.
   - Proposal: make it an interactive loop of generate → run on HiL → record pass/fail → next generation. Keep an archive of all tested samples with their verdicts.

3. **The goal "narrow down the scope of the error" is a boundary-finding problem.**
   - With pass and fail points, the failure-region boundary can be found more efficiently than by local mutation.
   - Options:
     - bisection between nearest pass/fail pairs
     - a cheap classifier (e.g. SVM or k-NN) and sampling near its decision boundary
     - Bayesian optimisation
   - The EA can stay as the exploration engine, but the fitness should reward points near the pass/fail boundary.

4. **The weights c > b > a do not mean what the README claims.**
   - V1 is scaled by 1/1000 but V2 is not. A t_End only 10 ms outside T gives c·V2 = 50, while V1 is typically about 1–5.
   - The real trade-off is c versus b/1000, so "c > b > a" is meaningless.
   - Proposal: normalise both terms by |T| into [0, 1], or treat T as a hard constraint (repair or clip) instead of a penalty.

5. **Roulette on 1/f is unstable.**
   - Every f = 0 becomes `mindestwert` = 1e-4, which gives a weight of 10 000. A sample sitting on an error point then monopolises selection.
   - With E empty, every valid sample gets f = 0, so selection is uniform and the search is a random walk.
   - The result depends heavily on the arbitrary `mindestwert`.
   - Proposal: use rank-based or tournament selection, or Boltzmann weights exp(−f/τ).

6. **The mutation step is too small for the search space.**
   - σ = 5 ms on a domain of about 5000 ms, and only 2 offspring × 4 generations, gives at most 8 new samples, all within about ±15 ms of their parents.
   - Proposal: use an adaptive σ (1/5 success rule or self-adaptation), or σ proportional to the distance to the nearest error. Optionally use a separate σ for t_start and t_End.

7. **The output should be a test *set*, but the fitness scores individuals.**
   - (μ+λ) elitism with popsize 4 collapses diversity. The final set clusters on one error, and the other errors in E are ignored.
   - Proposal: add a diversity term (crowding or niching per E_i), or allocate samples per error cluster.

8. **The representation and constraints are underspecified.**
   - Only t_End ∈ T is required. Can t_start lie before T_Start? Is a zero-length event allowed? The swap repair permits t_start == t_End.
   - `offset` lets the initial samples exceed T by ±10 ms, but that is undocumented.
   - Proposal: parameterise the event as (t_End, duration). The ordering constraint then disappears, and "t_End matters more" maps directly onto per-gene σ and weights. Also define a minimum duration.

9. **Traceability, given the ISO 26262 framing.**
   - There is no RNG seed and no log of generations or verdicts, so test generation cannot be reproduced.
   - Proposal: add a seed parameter and persist each generation together with its scores and results.

## B. README errors and inconsistencies
- The V2 formula is wrong for t_End > t_0. The code uses t_End − t_0; the README says |t_End − T_Start| in all "otherwise" cases. It should be "the distance to the nearest bound of T".
- The selection index notation selects element k+1 while describing k.
- Steps the README never describes:
  - survivor selection ((μ+λ), keep the best `popsize`)
  - the termination criterion (a fixed `step_2_iterations`)
  - the duplicate-removal step
  - the absence of recombination
  - the parameter table and the Streamlit app
- There is no "how to run" section.
- Typos: goverment, descripted, milisecons, weigthed, lowerest, popzise, "test's parameters".

## C. Code issues found while cross-checking
- `mutate_selected_offspring` (**solved**):
  - If every offspring is a duplicate, `offspring` becomes an empty 1-D array, and `fitness_function` / `V_1` crash on `[:, None, :]`.
    - Fix: boolean indexing keeps the (n, 2) shape, also when n = 0.
  - Duplicates are checked only against the current population, not against each other or against history.
    - Fix: `np.unique` removes duplicates within the offspring. `EA()` keeps a history of tested samples (`error_hit_set` plus every population so far) and passes it to `step_2` and the mutation.
- (**ok**) `step_1()` returns hard-coded samples. `initialize_samples_array()` is commented out.

- (**solved**) The algorithm module is duplicated between the root and WebApp/. Both copies rely on mutable module globals that the Streamlit UI patches.

- (**solved**) In the WebApp, "Run EA" before "Step 1" runs on an empty array, and the error is hidden by a bare `except`.

## Proposed follow-up (once the user picks a scope)
1. Fix the README (section B) and document the missing steps and parameters.
2. Fix the code bugs (section C) and remove the duplicate module so the WebApp imports from the root.
3. Concept changes, chosen from section A. The recommended minimum is:
   - normalised fitness (A4)
   - rank or tournament selection (A5)
   - a tested-sample archive with pass/fail fed back into the loop (A2, which enables A1 and A3)
   - a seed (A9)

## Verification (for any implementation)
- Run `python Evolutionary_Algorithm.py` with E empty and with the example E `[[-4025,-300],[-3213,-1232]]`.
  - Check that there is no crash when every offspring is a duplicate.
  - Check that the output stays within T.
  - Check that it is reproducible with a fixed seed.
- Run `streamlit run WebApp/layout.py` and click through Step 1 → Step 2, including running Step 2 before Step 1.
