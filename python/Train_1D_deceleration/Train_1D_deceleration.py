import json
from pathlib import Path

from simulation import simulate
from Train_1D_deceleration.plot import plot_all

ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "02_1D_Train_deceleration.json"
config_path = ROOT / "configs" / "02_1D_Train_deceleration.json"

def main():
    # Read scenario and filter config
    with open(scenario_path) as f:
        scenario = json.load(f)

    dt_s = scenario["sensors"]["GPS"]["dt_s"]
    sigma_m = scenario["sensors"]["GPS"]["sigma_m"]
    v_0 = scenario["train"]["v_0"]
    deceleration = scenario["train"]["deceleration"]
    seed = scenario.get("seed", None)

    t, s_true, v_true, s_measured, v_diff = simulate(scenario, seed)

    plot_all(t, s_true, s_measured, v_true, v_diff, sigma_m)