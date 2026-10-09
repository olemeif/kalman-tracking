import json
from pathlib import Path
import numpy as np

from simulation import simulate
from Train_1D.kalman_2nd_degree import kalman_2nd_degree
from Train_1D.plot import plot_all
from helpers import rmse


ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "01_1D_Train.json"
config_path = ROOT / "configs" / "01_1D_Train.json"

def main():
    # Read scenario and filter config
    with open(scenario_path) as f:
        scenario = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    dt_s = scenario["sensors"]["GPS"]["dt_s"]
    sigma_m = scenario["sensors"]["GPS"]["sigma_m"]
    v_0 = scenario["train"]["v_0"]
    seed = scenario.get("seed", None)

    t, s_true, v_true, s_measured, v_diff = simulate(scenario, seed)
    gps_noise = s_measured - s_true
    print(f"Noise: mean = {np.mean(gps_noise):.4f} m, std = {np.std(gps_noise):.4f} m")

    x_est, P_est, _ = kalman_2nd_degree(s_measured, dt_s, sigma_m, len(t), config["filter"]["v_0_prior"], config["filter"]["Q"])
    s_est, v_est = x_est[:, 0], x_est[:, 1]
    P_s, P_v = P_est[:, 0, 0], P_est[:, 1, 1]

    # RMSE Evaluation
    print(f"Position RMSE: measured = {rmse(s_measured - s_true):.2f} m, "
          f"filter = {rmse(s_est - s_true):.2f} m")
    print(f"Speed RMSE:    finite diff = {rmse(v_diff - v_true):.2f} m/s, "
          f"filter = {rmse(v_est - v_true):.2f} m/s")
    print(f"Final speed estimate: {v_est[-1]:.2f} m/s (true: {v_0:.2f} m/s), "
          f"2-sigma = {2 * np.sqrt(P_v[-1]):.2f} m/s")

    plot_all(t, s_true, s_measured, s_est, P_s, v_true, v_diff, v_est, P_v, sigma_m)
