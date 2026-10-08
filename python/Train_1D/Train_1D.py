import json
from pathlib import Path
import numpy as np

from Train_1D.simulation import simulate_ground_truth, generate_measurement
from Train_1D.kalman_2nd_degree import kalman_2nd_degree
from Train_1D.plot import plot_all
from helpers import rmse


ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "01_1D_Train.json"

def main():
    # Read scenario
    with open(scenario_path) as f:
        scenario = json.load(f)

    dt_s = scenario["sensors"]["GPS"]["dt_s"]
    sigma_m = scenario["sensors"]["GPS"]["sigma_m"]
    s_0 = scenario["train"]["s_0"]
    v_0 = scenario["train"]["v_0"]
    track_length_m = scenario["track_length_m"]
    seed = scenario.get("seed", None)

    n_steps, t, s_true, v_true = simulate_ground_truth(track_length_m, v_0, s_0, dt_s)
    s_measured, v_diff = generate_measurement(s_true, sigma_m, dt_s, n_steps, seed)

    s_est, v_est, P_s, P_v, K_s, K_v = kalman_2nd_degree(s_measured, dt_s, sigma_m, n_steps, 40.0)

    # RMSE Evaluation
    print(f"Position RMSE: measured = {rmse(s_measured - s_true):.2f} m, "
          f"filter = {rmse(s_est - s_true):.2f} m")
    print(f"Speed RMSE:    finite diff = {rmse(v_diff - v_true):.2f} m/s, "
          f"filter = {rmse(v_est - v_true):.2f} m/s")
    print(f"Final speed estimate: {v_est[-1]:.2f} m/s (true: {v_0:.2f} m/s), "
          f"2-sigma = {2 * np.sqrt(P_v[-1]):.2f} m/s")
    
    plot_all(t, s_true, s_measured, s_est, P_s, v_true, v_diff, v_est, P_v, sigma_m)
