import json
from pathlib import Path
import numpy as np

from simulation import simulate
from Train_1D_deceleration.kalman_3rd_degree import kalman_3rd_degree
from Train_1D_deceleration.plot import plot_all
from helpers import rmse

ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "02_1D_Train_deceleration.json"
config_path = ROOT / "configs" / "02_1D_Train_deceleration.json"

def main():
    # Read scenario and filter config
    with open(scenario_path) as f:
        scenario = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    dt_s = scenario["sensors"]["GPS"]["dt_s"]
    sigma_m = scenario["sensors"]["GPS"]["sigma_m"]
    seed = scenario.get("seed", None)

    t, s_true, v_true, a_true, s_measured, v_diff = simulate(scenario, seed)

    x_est, P_est, K_est = kalman_3rd_degree(
        s_measured=s_measured,
        dt_s=dt_s,
        sigma_m=sigma_m,
        v_0=config["filter"]["v_0_prior"],
        Q=config["filter"]["Q"]
    )
    s_est, v_est, a_est = x_est[:, 0], x_est[:, 1], x_est[:, 2]
    P_s, P_v, P_a = P_est[:, 0, 0], P_est[:, 1, 1], P_est[:, 2, 2]

    # Acceleration has no sensor. The only baseline is the finite difference of the already
    # noisy finite-difference speed, which amplifies the position noise twice.
    a_diff = np.full(v_diff.shape, np.nan)
    a_diff[1:] = np.diff(v_diff) / dt_s

    # RMSE Evaluation
    print(f"Position RMSE:     measured = {rmse(s_measured - s_true):.2f} m, "
          f"filter = {rmse(s_est - s_true):.2f} m")
    print(f"Speed RMSE:        finite diff = {rmse(v_diff - v_true):.2f} m/s, "
          f"filter = {rmse(v_est - v_true):.2f} m/s")
    print(f"Acceleration RMSE: finite diff = {rmse(a_diff - a_true):.2f} m/s², "
          f"filter = {rmse(a_est - a_true):.2f} m/s²")

    plot_all(t, s_true, s_measured, s_est, P_s, v_true, v_diff, v_est, P_v, a_true, a_est, P_a, sigma_m)
