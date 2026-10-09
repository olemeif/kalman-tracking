import json
from pathlib import Path

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