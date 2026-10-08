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

def plot_monte_carlo(t, results, bounds, n_runs):
    """
    Monte Carlo evaluation, one curve per process noise value Q.
    results: dict Q -> (rmse_t, sigma_t, anees_t), each averaged across runs per time step.
    """
    fig, (ax_s, ax_v, ax_a) = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (Q, (rmse_t, sigma_t, anees_t)), color in zip(results.items(), colors):
        for ax, i in ((ax_s, 0), (ax_v, 1)):
            ax.plot(t, rmse_t[:, i], color=color, label=f"RMSE, Q = {Q:g}")
            ax.plot(t, sigma_t[:, i], color=color, ls="--", label=f"Filter σ, Q = {Q:g}")
        ax_a.plot(t, anees_t, color=color, label=f"Q = {Q:g}")

    r1, r2 = bounds
    ax_a.axhspan(r1, r2, color="gray", alpha=0.2, label=f"95% bounds [{r1:.2f}, {r2:.2f}]")
    ax_a.axhline(2, color="black", lw=1, ls=":")

    ax_s.set_ylabel("Position error (m)")
    ax_s.set_title("Position: RMSE vs. filter σ")
    ax_v.set_ylabel("Speed error (m/s)")
    ax_v.set_title("Speed: RMSE vs. filter σ")
    ax_a.set_ylabel("ANEES")
    ax_a.set_title("ANEES ([1], Eq. 9.12)")
    ax_a.set_ylim(0, 4)
    for ax in (ax_s, ax_v):
        ax.set_yscale("log")
    for ax in (ax_s, ax_v, ax_a):
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()

    fig.suptitle(f"Stage 1: Monte Carlo evaluation, {n_runs} runs")
    fig.tight_layout()
    plt.show()
