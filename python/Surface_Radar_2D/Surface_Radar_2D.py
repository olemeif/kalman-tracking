import json
from pathlib import Path
import numpy as np

from simulation import simulate
from Surface_Radar_2D.kalman import kalman
from Surface_Radar_2D.plot import plot_simulation
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

    seed = scenario.get("seed", None)

    # Simulate the scenario
    t, s_true, v_true, a_true, s_measured, v_diff = simulate(scenario, seed, "turns")

    # Plot the simulation results
    plot_simulation(t, s_true, v_true, a_true, s_measured, v_diff)