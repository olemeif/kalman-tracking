# kalman-tracking
A learning project on state estimation and target tracking. It starts with a 1D Kalman filter for a train on a straight track and builds step by step towards a 2D/3D multi-target radar tracker with systematic performance evaluation.

The filters are developed in Python first and then ported to C++. Python remains the layer for simulation, evaluation and plotting.

## Setup (Python)

Requires Python 3.13 or newer.

Clone the repository then navigate to the project directory, create a virtual environment and install the dependencies:

```sh
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

`pip install -e .` installs the dependencies and makes the shared modules in `python/` (e.g. `helpers`) importable from every stage. Then run a stage from the `python/` directory, e.g.:

```sh
cd python
python main.py                  # single run with the scenario's seed
python main.py --monte-carlo    # Monte Carlo evaluation over many seeds
```

If you get `ModuleNotFoundError: No module named 'helpers'`, the virtual environment is not active or `pip install -e .` has not been run.

## Stage 1: 1D Kalman filter for a train on a straight track

A train moves along a straight track of length $\ell$ at a constant speed $v_0$. A position sensor measures the train's position every $T_s$ seconds with Gaussian noise. A Kalman filter with a 2nd-order kinematic model (constant-velocity model, [1], Sec. 12.2) estimates position **and** speed, although the speed is never measured.
 
### Model
 
```math
\mathbf{x}(k) = \begin{bmatrix} s(k) \\ v(k) \end{bmatrix}, \qquad
\mathbf{A}_d = \begin{bmatrix} 1 & T_s \\ 0 & 1 \end{bmatrix}, \qquad
\mathbf{C} = \begin{bmatrix} 1 & 0 \end{bmatrix}, \qquad
R = \sigma^2
```
 
([1], Eq. 12.3, 12.4 and 1.18). Filter equations: [1], Eq. 12.24–12.28, in the order *correct with measurement k, then predict to k+1*.
 
**Process noise.** [1], Sec. 12.2 gives three ways to discretize it. All three are implemented, written with $\text{Var}(z_v)$, the velocity change per step:
 
**Method 1: direct discretization** ([1], Eq. 12.11)
 
```math
\mathbf{G}_d \mathbf{Q} \mathbf{G}_d^T = \begin{bmatrix} T_s^2 & T_s \\ T_s & 1 \end{bmatrix} \text{Var}(z_v)
```
 
**Method 2: piecewise constant noise** ([1], Eq. 12.17), used for the results below
 
```math
\mathbf{G}_d \mathbf{Q} \mathbf{G}_d^T = \begin{bmatrix} T_s^2/4 & T_s/2 \\ T_s/2 & 1 \end{bmatrix} \text{Var}(z_v)
```
 
**Method 3: discretized continuous model** ([1], Eq. 12.19)
 
```math
\mathbf{G}_d \mathbf{Q} \mathbf{G}_d^T = \begin{bmatrix} T_s^2/3 & T_s/2 \\ T_s/2 & 1 \end{bmatrix} \text{Var}(z_v)
```
 
### Parameters
 
| Scenario | Value |
|---|---|
| Track length $\ell$ | 2,500 m |
| Initial position $s_0$ | 0 m |
| Speed $v_0$ | 55.55 m/s (200 km/h) |
| Sampling time $T_s$ | 1.0 s |
| Measurement noise $\sigma$ | 5.0 m |
| Random seed | 31 |
 
| Filter | Value |
|---|---|
| Measurement noise $R$ | $\sigma^2$ = 25 m² (matches the simulation) |
| Process noise $\text{Var}(z_v)$ | 0.01 (m/s)², method 2 (Eq. 12.17) |
| Initial state $\hat{\mathbf{x}}_0$ | $[s_{\text{meas}}(0),\ 40\ \text{m/s}]^T$ (speed deliberately wrong) |
| Initial covariance $\hat{\mathbf{P}}_0$ | $\text{diag}(100^2\ \text{m}^2,\ 20^2\ (\text{m/s})^2)$ |

### Results

![1D Kalman filter for a train on a straight track](docs/assets/images/01_1D_Train.png)

The speed baseline is the finite difference of consecutive position measurements, $(y(k) - y(k-1))/T_s$. RMSE over a single run (seed 31):
 
| RMSE | Raw sensor / finite difference | Kalman filter |
|---|---|---|
| Position, all steps | 4.56 m | 2.29 m |
| Position, after convergence ($k \geq 5$) | 4.70 m | 2.34 m |
| Speed, all steps | 6.32 m/s | 2.34 m/s |
| Speed, after convergence ($k \geq 5$) | 6.33 m/s | 0.26 m/s |

**Observations**
 
- **Speed without a speed sensor.** Starting from 40 m/s (true: 55.55 m/s), the estimate converges within about 3 s. The overall speed RMSE is dominated by this start-up; after convergence the filter is about 25× more accurate than the finite difference.
- **Position.** The filter roughly halves the position error. Its ±2σ band shrinks from the sensor's ±10 m to about ±4.3 m.

These numbers come from one noise realization only. Whether the ±2σ bands are actually right (filter consistency) needs many runs; see the Monte Carlo evaluation below.

### Monte Carlo evaluation

The same experiment is repeated $N = 500$ times with seeds $0, \dots, N-1$. Errors are averaged **across runs, not across time**, which gives one value per time step and shows how the filter converges.

- **RMSE vs. filter σ.** The RMSE over all runs is compared with the filter's own claim, $\sqrt{\tilde{P}_{ii}}$. For a consistent filter, the two curves lie on top of each other.
- **ANEES** ([1], Eq. 9.7 and 9.12). For each run and step, $\varepsilon(k) = \tilde{\boldsymbol{\varepsilon}}^T \tilde{\mathbf{P}}^{-1} \tilde{\boldsymbol{\varepsilon}}$ uses the full covariance, because position and speed errors are correlated. Averaged over $N$ runs, $N \cdot \bar{\varepsilon}$ follows a $\chi^2$ distribution with $N \cdot n$ degrees of freedom ($n = 2$ states). The 95% bounds are $[r_1, r_2] = [1.83,\ 2.18]$ ([1], Eq. 9.13). A consistent filter stays around $n = 2$, inside the bounds.

Two process noise values are compared: the value from the single run, $\text{Var}(z_v) = 0.01$, and $\text{Var}(z_v) = 0$, which matches the simulation exactly (the speed never changes).

![Monte Carlo evaluation of the 1D Kalman filter](docs/assets/images/01_1D_Train_MC.png)

Averaged over the second half of the run ($t \geq 23$ s):

| $\text{Var}(z_v)$ | Position RMSE / filter σ | Speed RMSE / filter σ | ANEES | Steps inside bounds |
|---|---|---|---|---|
| 0.01 (m/s)² | 1.88 m / 2.14 m | 0.157 m/s / 0.309 m/s | 1.03 | 15% |
| 0 | 1.67 m / 1.68 m | 0.089 m/s / 0.090 m/s | 1.94 | 98% |

**Observations**

- **$\text{Var}(z_v) = 0.01$ is too pessimistic.** The filter allows for speed changes that never happen. Its σ levels off while the actual error keeps falling, so it claims about twice the actual speed error. The ANEES drops below the lower bound at about 9 s and ends near 1.0.
- **$\text{Var}(z_v) = 0$ is consistent.** RMSE and σ coincide for both states, and the ANEES stays around 2 throughout. It is also the more accurate filter: the model matches the simulation, so every measurement keeps refining the speed.
- **Start-up.** Both filters agree for the first ~10 s, where the large initial covariance $\hat{\mathbf{P}}_0$ dominates. At $k = 0$, the ANEES is about 1.5, because $\hat{\mathbf{P}}_0$ is deliberately generous.
- **Why not use 0 in practice?** With $\text{Var}(z_v) = 0$, the gain goes to zero and the filter could never follow a real speed change. In practice, a small $\text{Var}(z_v) > 0$ is chosen, accepting a slightly conservative (but safe) filter. An *overconfident* filter, with the σ curve below the RMSE, would be the dangerous case.

The 46-step track is short, so neither filter reaches a true steady state; with $\text{Var}(z_v) = 0$, the covariance keeps shrinking until the end.

## Next steps

- Create Test Cases
- Implement 3rd Order Model
- Scenario with real speed changes, to find a $\text{Var}(z_v)$ that is both responsive and consistent
- Compare the three process noise methods
- Effect of a too small $\hat{\mathbf{P}}_0$ on convergence

## Notation
 
This project follows the notation of [1]:
 
| Symbol | Meaning |
|---|---|
| $s$, $v$ | position, velocity |
| $\mathbf{A}_d$, $\mathbf{B}_d$ | discrete system and input matrix |
| $\mathbf{C}$ | measurement matrix (often $\mathbf{H}$ in English literature) |
| $y$ | measurement |
| $\hat{\mathbf{x}}$, $\hat{\mathbf{P}}$ | **predicted** state and covariance |
| $\tilde{\mathbf{x}}$, $\tilde{\mathbf{P}}$ | **corrected** state and covariance |
| $\mathbf{Q}$, $\mathbf{R}$ | process and measurement noise covariance |
 
> [!NOTE]
> Many English sources use $\hat{\mathbf{x}}$ for the *corrected* estimate.
 
## References
 
[1] R. Marchthaler, S. Dingler: *Kalman-Filter. Einführung in die Zustandsschätzung und ihre Anwendung für eingebettete Systeme*, 2nd ed., Springer Vieweg, 2024. https://doi.org/10.1007/978-3-658-43216-4
