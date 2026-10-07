import json
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np


ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "01_1D_Train.json"

# Read scenario
with open(scenario_path) as f:
    scenario = json.load(f)

# Generate ground truth data
# Get n_steps
n_steps = int(scenario["track_length"]/(scenario["train"]["v_0"]*scenario["sensors"]["GPS"]["dt_s"]))
# Get position vector
t = np.arange(n_steps) * scenario["sensors"]["GPS"]["dt_s"]
s_true = scenario["train"]["s_0"] + scenario["train"]["v_0"] * t
v_true = np.full(n_steps, scenario["train"]["v_0"])

# Set random seed for reproducibility
seed = scenario.get("seed", None)
rng = np.random.default_rng(seed)

# Add noise to the ground truth data
sigma_m = scenario["sensors"]["GPS"]["sigma_m"]
gps_noise = rng.normal(0, sigma_m, size=s_true.shape)
s_measured = s_true + gps_noise

# Calculate Standard Deviation of the noise
std_dev = np.std(gps_noise)
print(f"Standard Deviation of the noise: {std_dev:.4f} m")

# Plot
fig, (ax, ax2) = plt.subplots(1, 2)
ax.plot(t, s_true, label="Ground Truth", color="blue")
ax.scatter(t, s_measured, label="Measured", color="red", s=10)
ax.set_xlabel("Time step")
ax.set_ylabel("Position (m)")
ax.set_title("1D Train Position: Ground Truth vs Measured")
ax.legend()
ax2.scatter(t, s_measured - s_true, s=10)
ax2.axhline(2 * sigma_m, ls="--")
ax2.axhline(-2 * sigma_m, ls="--")
plt.show()
