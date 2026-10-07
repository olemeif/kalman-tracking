import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]   # adjust to your folder depth
scenario_path = ROOT / "scenarios" / "01_1D_Train.json"

# Read scenario
with open(scenario_path) as f:
    scenario = json.load(f)

dt_s = scenario["sensors"]["GPS"]["dt_s"]

# Generate ground truth data
# Get n_steps
n_steps = int(scenario["track_length"]/(scenario["train"]["v_0"]*dt_s)) + 1
# Get position vector
t = np.arange(n_steps) * dt_s
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

# Speed estimation using finite difference
v_diff = np.full(n_steps, np.nan)
v_diff[1:] = np.diff(s_measured) / dt_s

R = sigma_m ** 2 # [1, Eq. 1.18]
Q = 0.01 # Var(z_v) in (m/s)^2 per step, Eyeballed process noise, since we don't have a model for the train's acceleration

A_d = np.array([[1, dt_s], [0, 1]]) # [1, Eq. 12.4]
C = np.array([[1, 0]]) # [1, Eq. 12.3]

# GQG = np.array([[dt_s**2,     dt_s],     [dt_s,     1]]) * Q # [1, Eq. 12.11]
GQG   = np.array([[1/4*dt_s**2, 1/2*dt_s], [1/2*dt_s, 1]]) * Q # [1, Eq. 12.17]
# GQG = np.array([[1/3*dt_s**2, 1/2*dt_s], [1/2*dt_s, 1]]) * Q # [1, Eq. 12.19]

x_hat = np.array([[s_measured[0]], [40.0]])     # initial state estimate
P_hat = np.array([[100.0**2, 0], [0, 20.0**2]]) # initial covariance estimate

s_est = np.zeros(n_steps)   # state estimate vector
v_est = np.zeros(n_steps)   # velocity estimate vector
P_s = np.zeros(n_steps)     # Position covariance estimate vector
P_v = np.zeros(n_steps)     # Velocity covariance estimate vector
K_s = np.zeros(n_steps)     # Kalman gain, position component
K_v = np.zeros(n_steps)     # Kalman gain, velocity component

# Kalman Filter
for k in range(n_steps):
    K = P_hat @ C.T @ np.linalg.inv(C @ P_hat @ C.T + R)    # [1, Eq. 12.24]
    x_tilde = x_hat + K @ (s_measured[k] - C @ x_hat)       # [1, Eq. 12.25]
    P_tilde = (np.eye(2) - K @ C) @ P_hat                   # [1, Eq. 12.26]

    # Store the corrected estimate for time k
    s_est[k] = x_tilde[0, 0]
    v_est[k] = x_tilde[1, 0]
    P_s[k] = P_tilde[0, 0]
    P_v[k] = P_tilde[1, 1]
    K_s[k] = K[0, 0]
    K_v[k] = K[1, 0]

    # Prediction for time k+1
    x_hat = A_d @ x_tilde # [1, Eq. 12.27]
    P_hat = A_d @ P_tilde @ A_d.T + GQG # [1, Eq. 12.28]

# Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
(ax_s, ax_se), (ax_v, ax_ve) = axs

# Position
ax_s.plot(t, s_true, label="Ground truth", color="black")
ax_s.scatter(t, s_measured, label="Measured", color="tab:red", s=12)
ax_s.plot(t, s_est, label="Kalman filter", color="tab:blue")
ax_s.set_ylabel("Position (m)")
ax_s.set_title("Position")
ax_s.legend()

# Position error
band_s = 2 * np.sqrt(P_s)
ax_se.scatter(t, s_measured - s_true, label="Measured", color="tab:red", s=12)
ax_se.plot(t, s_est - s_true, label="Kalman filter", color="tab:blue")
ax_se.fill_between(t, -band_s, band_s, color="tab:blue", alpha=0.15, label="Filter ±2σ")
ax_se.axhline(2 * sigma_m, ls="--", color="tab:red", lw=1, label="Sensor ±2σ")
ax_se.axhline(-2 * sigma_m, ls="--", color="tab:red", lw=1)
ax_se.set_ylabel("Error (m)")
ax_se.set_title("Position error")
ax_se.set_ylim(-4 * sigma_m, 4 * sigma_m)  # first steps have a large band
ax_se.legend()

# Speed
band_v = 2 * np.sqrt(P_v)
ax_v.plot(t, v_true, label="Ground truth", color="black")
ax_v.scatter(t, v_diff, label="Finite difference", color="tab:red", s=12)
ax_v.plot(t, v_est, label="Kalman filter", color="tab:blue")
ax_v.fill_between(t, v_est - band_v, v_est + band_v, color="tab:blue", alpha=0.15,
                  label="Filter ±2σ")
ax_v.set_xlabel("Time (s)")
ax_v.set_ylabel("Speed (m/s)")
ax_v.set_title("Speed")
ax_v.legend()

# Speed error
ax_ve.scatter(t, v_diff - v_true, label="Finite difference", color="tab:red", s=12)
ax_ve.plot(t, v_est - v_true, label="Kalman filter", color="tab:blue")
ax_ve.fill_between(t, -band_v, band_v, color="tab:blue", alpha=0.15, label="Filter ±2σ")
ax_ve.set_xlabel("Time (s)")
ax_ve.set_ylabel("Error (m/s)")
ax_ve.set_title("Speed error")
ax_ve.legend()

for ax in axs.flat:
    ax.grid(alpha=0.3)

fig.suptitle("Stage 1: 1D train, constant velocity, position sensor only")
fig.tight_layout()
plt.show()
