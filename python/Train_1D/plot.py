import matplotlib.pyplot as plt
import numpy as np

def plot_all(t, s_true, s_measured, s_est, P_s, v_true, v_diff, v_est, P_v, sigma_m):
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