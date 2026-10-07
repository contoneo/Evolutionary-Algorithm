### Introduction
This algorithm was presented as a part in the thesis I wrote for my Master of Science (M.Sc.) degree in the “Automotive Software Engineering” program, whose main objective was to provide a testing methodology based on the ISO 26262 standard to address the new regulations taking effect in 2021 for EDRs and DSSADs within the framework of several markets around the world.  EDR (Event Data Recorder) and DSSAD (Data Storage System for Automated Driving) are devices installed in vehicles to record critical driving and crash dynamics data during an accident.

The proposed approach to create test samples assumes the discovery of a malfunction during a test.  Thereafter, it can be used to exploit the flaw, uncovering other related malfunctions, and narrowing down the scope and behavior of the error within the limits of the HiL test.

### Description
In the operation of an EDR, there are different triggers for activating  and recording an event, these are defined in goverment resolutions.  The test under consideration examines whether an activation trigger might prevent the recording of an event, depending on the timing of each trigger.

If some errors are detected during a HIL test cycle for the case above descripted, further test's parameters can be optimized by this Evolutionary Algorithm (hereinafter referred to simply by the acronym **EA**).

![](EDR_activity_state.png)

*EDR test case for which the **EA** is made*

#### Population

A test set for this test case (**EA**'s output) is a collection of tuples setting the start and the end of the so called "trivial event"  in the way
$$(t_{\text{start}}, t_{\text{End}})$$
where 
* $(t_{\text{start}} < t_{\text{End}}$ and $t_{\text{End}} \in T)$
* $T=$  Pre-crash recording time established in the regulation.

#### Fitness Orientation

The optimization made by the **EA** works according to the principles of software testing "defect clustering" and "the Pesticide Paradox":
* Defect clustering: Most software bugs are concentrated within a small number of complex or heavily used modules.
* The Pesticide Paradox: Repeatedly running the same tests will eventually stop finding new bugs as the software evolves.

Then, the fitness function $f$ for a tuple $(t_{\text{start}}, t_{\text{End}})$ with respect to all *i*_tuples in $E$ is expressed as:
* $\displaystyle f=V_1 +c⋅V_2$, Where
* $\displaystyle V_1 = \frac{a \cdot \min\limits_{i \in E}\Big|t_{\text{start}} - E_i(t_{\text{start}})\Big| + b \cdot \min\limits_{j \in E}\Big|t_{\text{End}} - E_j(t_{\text{End}})\Big|}{1000}$
* $\displaystyle V_2 = \begin{cases} 0 & \text{if } t_{\text{End}} \in T \\ |t_{\text{End}} - T_{\text{Start}}| & \text{otherwise} \end{cases}$

The $V_1$ equation can be read as: "The distance from $t_{\text{start}}$ to the closest start time of any tuple in the captured malfunction's collection $E$, weighted by $a$, plus the distance from $t_{\text{End}}$ to the closest end time of any tuple in $E$, weighted by $b$, divided by 1000", where:

* $E$: Set of test samples that have led to find an error. Consequently, $E_i(t_{\text{start}})$ and $E_j(t_{\text{End}})$ are the first and second component of a tuple in the collection $E$.
* $t_{\text{Start}}$ and $t_{\text{End}}$ are chosen independently, so their closest values may come from different tuples in $E$. This allows the algorithm to search any combination of known start and end times, rather than requiring a proposed tuple to match a single known tuple.
* As the tuples are given in milliseconds, the weighted distance is divided by $1000$ so that the result can be managed in one digit.
* The end of the examined trivial event ($t_{\text{End}}$) is more important than its beginning ($t_{\text{Start}}$), therefore the equation prioritizes<br>$\min\limits_{j \in E}\Big|t_{\text{End}} - E_j(t_{\text{End}})\Big|$ over $\min\limits_{i \in E}\Big|t_{\text{start}} - E_i(t_{\text{start}})\Big|$. Thus $b$ must be accordingly higher than $a$ ($b>a$).
* If $E$ is empty, then $V_1=0$ consequently.

The $V_2$ equation scores lower the trivial events (*the tuples* $(t_{\text{start}}, t_{\text{End}})$) that ends within the pre-crash recording time established by a market regulation, as the main purpose of the HiL test is to verify the functioning of an EDR device within the parameters established by a market 
regulation. If the tuple doesn't belong to the Pre-crash recording time established in the regulation ($T$) then the value for this function is the distance to the that range start.
Therefore:
* $c$ in $f$ function must be greater than $a$ and $b$, therefore: $c>b>a$

The best scored tuple $(t_{\text{start}}, t_{\text{End}})$ by function $f$ is the lowerest scored.

#### Variation Drive
The **EA** uses the “Roulette wheel selection” (RWS) method for selection.
Then relative fitness value for a tuple $i$ is:

$$p_i = \frac{f_i^{-1}}{\displaystyle\sum_{i=1}^{\text{popsize}} f_i^{-1}}, \text{ where, } \sum_{i=1}^{\text{popsize}} p_i = 1$$

* *popzise*: (Population size) Is the number of samples created before its selection.
* As the best posible score in $f$ function tends to zero, the prior equation uses $f_i^{-1}$ instead of $f_i$

Then a tuple $i$ will be selected by $$\sum_{i=1}^{k} p_i < \text{rand} < \sum_{i=1}^{k+1} p_i$$
* where $\text{rand}$ is a random number between [0,1]

#### Mutation
The *EA* uses “normal mutation”, which performs the mutation on a parameter i of a tuple :
$$x_i = x_i + N(0, \sigma)$$