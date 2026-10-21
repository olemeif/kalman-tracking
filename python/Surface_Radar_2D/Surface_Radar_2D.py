import json
from pathlib import Path
import numpy as np

from simulation import simulate
from Surface_Radar_2D.kalman import kalman
from Surface_Radar_2D.plot import plot_all
from helpers import rmse

ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "03_2D_Surface_Radar.json"
config_path = ROOT / "configs" / "03_2D_Surface_Radar.json"

def main():
    # Load scenario and configuration
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)

    dt_s = scenario["sensors"]["SRA"]["dt_s"]
    sigma_m = scenario["sensors"]["SRA"]["sigma_m"]
    seed = scenario.get("seed", None)
    target = "turns"
    maneuvers = scenario["targets"][target].get("maneuvers", [])

    # Simulate the scenario
    t, s_true, v_true, a_true, s_measured, v_diff = simulate(scenario, seed, target)

    # Kalman filter with constant Q on the same measurements
    x_kf, P_kf, K_kf = kalman(
        s_measured=s_measured,
        dt_s=dt_s,
        sigma_m=sigma_m,
        v_0=config["filter"]["v_0_prior"],
        Q=config["filter"]["Q"]
    )
    s_kf, v_kf = x_kf[:, :2], x_kf[:, 2:]

    # RMSE Evaluation
    print(f"Position RMSE: measured = {rmse(s_measured - s_true):.2f} m, "
          f"filter = {rmse(s_kf - s_true):.2f} m")
    print(f"Velocity RMSE: finite diff = {rmse(v_diff - v_true):.2f} m/s, "
          f"filter = {rmse(v_kf - v_true):.2f} m/s")

    plot_all(t, s_true, s_measured, s_kf, v_true, v_diff, v_kf, P_kf, sigma_m, dt_s, maneuvers)
