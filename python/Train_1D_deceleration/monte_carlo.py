import json
from pathlib import Path
import numpy as np

from simulation import simulate
from Train_1D_deceleration.kalman_3rd_degree import kalman_3rd_degree
from Train_1D_deceleration.plot import plot_monte_carlo
from helpers import nees, anees_bounds, rmse


ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "02_1D_Train_deceleration.json"
config_path = ROOT / "configs" / "02_1D_Train_deceleration.json"

N_STATES = 3   # [s, v, a]

def simulate_runs(scenario: dict, n_runs: int):
    """
    Simulate n_runs noise realizations of the scenario (seeds 0 ... n_runs-1).
    The ground truth is deterministic, only the measurement noise differs between runs.
    Returns:
        t: np.ndarray, time vector (n_steps,)
        x_true: np.ndarray, true state [s, v, a] (n_steps, 3)
        s_measured: np.ndarray, measured positions (n_runs, n_steps)
        v_diff: np.ndarray, finite-difference speeds (n_runs, n_steps)
    """
    t, s_true, v_true, a_true, _, _ = simulate(scenario, seed=0)
    s_measured = np.zeros((n_runs, len(t)))
    v_diff = np.zeros((n_runs, len(t)))
    for i in range(n_runs):
        s_measured[i], v_diff[i] = simulate(scenario, seed=i)[4:]
    return t, np.column_stack([s_true, v_true, a_true]), s_measured, v_diff

def run_filter(scenario: dict, s_measured: np.ndarray, x_true: np.ndarray, v_0_prior: float, Q: float):
    """
    Run the 3rd-order filter on every measurement series.
    Args:
        scenario: dict, scenario as read from the scenario JSON file
        s_measured: np.ndarray, measured positions (n_runs, n_steps)
        x_true: np.ndarray, true state [s, v, a] (n_steps, 3)
        v_0_prior: float, initial speed estimate of the filter
        Q: float, process noise Var(z_a) of the filter
    Returns:
        err: np.ndarray, estimation errors x_tilde - x for [s, v, a] (n_runs, n_steps, 3)
        P_est: np.ndarray, corrected covariances P_tilde (n_runs, n_steps, 3, 3)
        nees_k: np.ndarray, NEES per run and step (n_runs, n_steps)
    """
    dt_s = scenario["sensors"]["GPS"]["dt_s"]
    sigma_m = scenario["sensors"]["GPS"]["sigma_m"]

    n_runs, n_steps = s_measured.shape
    err = np.zeros((n_runs, n_steps, N_STATES))
    P_est = np.zeros((n_runs, n_steps, N_STATES, N_STATES))
    for i in range(n_runs):
        x_est, P_est[i], _ = kalman_3rd_degree(s_measured[i], dt_s, sigma_m, v_0_prior, Q)
        err[i] = x_est - x_true

    return err, P_est, nees(err, P_est)   # [1, Eq. 9.7]

def main():
    with open(scenario_path) as f:
        scenario = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    mc = config["monte_carlo"]
    n_runs = mc["n_runs"]
    k0 = mc["start_up_steps"]
    r1, r2 = anees_bounds(n_runs, N_STATES, mc["confidence"])
    print(f"{n_runs} runs, {mc['confidence']:.0%} ANEES bounds for n = {N_STATES}: [{r1:.2f}, {r2:.2f}]")
    print(f"Errors after start-up (k >= {k0}); the braking phase is included.\n")

    t, x_true, s_measured, v_diff = simulate_runs(scenario, n_runs)

    # Baseline without a filter: raw sensor and finite difference
    print(f"{'Filter':<28}{'Pos RMSE [m]':>14}{'Speed RMSE [m/s]':>18}{'Acc RMSE [m/s²]':>17}"
          f"{'ANEES max':>11}{'ANEES median':>14}")
    print(f"{'Raw sensor / finite diff':<28}{rmse((s_measured - x_true[:, 0])[:, k0:]):>14.2f}"
          f"{rmse((v_diff - x_true[:, 1])[:, k0:]):>18.2f}{'-':>17}{'-':>11}{'-':>14}")

    results = {}
    for Q in mc["Q_values"]:
        err, P_est, nees_k = run_filter(scenario, s_measured, x_true, config["filter"]["v_0_prior"], Q)

        # Average across runs (axis 0), not across time: one value per time step
        rmse_t = np.sqrt(np.mean(err ** 2, axis=0))                                # (n_steps, 3)
        sigma_t = np.sqrt(np.mean(np.diagonal(P_est, axis1=2, axis2=3), axis=0))   # (n_steps, 3)
        anees_t = np.mean(nees_k, axis=0)   # [1, Eq. 9.12]                         (n_steps,)
        results[Q] = (rmse_t, sigma_t, anees_t)

        e = rmse(err[:, k0:, 0]), rmse(err[:, k0:, 1]), rmse(err[:, k0:, 2])
        print(f"{f'3rd order, Var(z_a) = {Q:g}':<28}{e[0]:>14.2f}{e[1]:>18.2f}{e[2]:>17.2f}"
              f"{np.max(anees_t[k0:]):>11.2f}{np.median(anees_t[k0:]):>14.2f}")

    plot_monte_carlo(t, results, (r1, r2), n_runs, scenario["train"]["deceleration"], k0)
