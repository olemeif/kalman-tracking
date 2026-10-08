import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2

from Train_1D.simulation import simulate
from Train_1D.kalman_2nd_degree import kalman_2nd_degree
from Train_1D.plot import plot_monte_carlo
from helpers import nees


ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "01_1D_Train.json"
config_path = ROOT / "configs" / "01_1D_Train.json"

N_STATES = 2   # [s, v]

def run_monte_carlo(scenario: dict, v_0_prior: float, Q: float, n_runs: int):
    """
    Run the same experiment n_runs times with a different noise seed each time.
    Args:
        scenario: dict, scenario as read from the scenario JSON file
        v_0_prior: float, initial speed estimate of the filter
        Q: float, process noise Var(z_v) of the filter
        n_runs: int, number of Monte Carlo runs (seeds 0 ... n_runs-1)
    Returns:
        t: np.ndarray, time vector (n_steps,)
        err: np.ndarray, estimation errors x_tilde - x for [s, v] (n_runs, n_steps, 2)
        P_est: np.ndarray, corrected covariances P_tilde (n_runs, n_steps, 2, 2)
        nees_k: np.ndarray, NEES per run and step (n_runs, n_steps)
    """
    dt_s = scenario["sensors"]["GPS"]["dt_s"]
    sigma_m = scenario["sensors"]["GPS"]["sigma_m"]

    t = simulate(scenario, seed=0)[0]
    n_steps = len(t)
    err = np.zeros((n_runs, n_steps, N_STATES))
    P_est = np.zeros((n_runs, n_steps, N_STATES, N_STATES))

    for i in range(n_runs):
        _, s_true, v_true, s_measured, _ = simulate(scenario, seed=i)
        x_est, P_est[i], _ = kalman_2nd_degree(s_measured, dt_s, sigma_m, n_steps, v_0_prior, Q)
        err[i] = x_est - np.column_stack([s_true, v_true])

    nees_k = nees(err, P_est)   # [1, Eq. 9.7]
    return t, err, P_est, nees_k

def anees_bounds(n_runs: int, n_states: int, confidence: float):
    """
    Confidence interval for ANEES. N * ANEES is chi-squared with N * n degrees of freedom.
    [1, Eq. 9.13]
    """
    alpha = 1 - confidence
    r1 = chi2.ppf(alpha / 2, n_runs * n_states) / n_runs
    r2 = chi2.ppf(1 - alpha / 2, n_runs * n_states) / n_runs
    return r1, r2

def main():
    with open(scenario_path) as f:
        scenario = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    mc = config["monte_carlo"]
    n_runs = mc["n_runs"]
    r1, r2 = anees_bounds(n_runs, N_STATES, mc["confidence"])
    print(f"{n_runs} runs, {mc['confidence']:.0%} ANEES bounds: [{r1:.2f}, {r2:.2f}]")

    results = {}
    for Q in mc["Q_values"]:
        t, err, P_est, nees_k = run_monte_carlo(scenario, config["filter"]["v_0_prior"], Q, n_runs)

        # Average across runs (axis 0), not across time: one value per time step
        rmse_t = np.sqrt(np.mean(err ** 2, axis=0))                                # (n_steps, 2)
        sigma_t = np.sqrt(np.mean(np.diagonal(P_est, axis1=2, axis2=3), axis=0))   # (n_steps, 2)
        anees_t = np.mean(nees_k, axis=0)   # [1, Eq. 9.12]                         (n_steps,)
        results[Q] = (rmse_t, sigma_t, anees_t)

        # Steady state: second half of the run, after the start-up transient
        ss = slice(len(t) // 2, None)
        inside = np.mean((anees_t >= r1) & (anees_t <= r2))   # [1, Eq. 9.11]
        print(f"Q = {Q:g}: steady-state RMSE s = {np.mean(rmse_t[ss, 0]):.2f} m "
              f"(filter σ {np.mean(sigma_t[ss, 0]):.2f} m), "
              f"v = {np.mean(rmse_t[ss, 1]):.3f} m/s (filter σ {np.mean(sigma_t[ss, 1]):.3f} m/s), "
              f"ANEES = {np.mean(anees_t[ss]):.2f}, inside bounds: {inside:.0%} of steps")

    plot_monte_carlo(t, results, (r1, r2), n_runs)
