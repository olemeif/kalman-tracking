import matplotlib.pyplot as plt
import numpy as np

def plot_all(t, s_true, s_measured, s_est, P_s, v_true, v_diff, v_est, P_v, a_true, a_est, P_a, sigma_m):
    """
    Plot the results of the 1D train scenario.
    Args:
        t: np.ndarray, time vector
        s_true: np.ndarray, true position vector
        s_measured: np.ndarray, measured position vector with noise
        s_est: np.ndarray, estimated position vector from Kalman filter
        P_s: np.ndarray, estimated position variance from Kalman filter
        v_true: np.ndarray, true velocity vector
        v_diff: np.ndarray, velocity vector from finite difference of measured position
        v_est: np.ndarray, estimated velocity vector from Kalman filter
        P_v: np.ndarray, estimated velocity variance from Kalman filter
        a_true: np.ndarray, true acceleration vector
        a_est: np.ndarray, estimated acceleration vector from Kalman filter
        P_a: np.ndarray, estimated acceleration variance from Kalman filter
        sigma_m: float, standard deviation of the measurement noise
    """
    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(t, s_true, label='True Position', color='black')
    plt.scatter(t, s_measured, label='Measured Position', color='red', s=10)
    plt.plot(t, s_est, label='Estimated Position', color='blue')
    plt.fill_between(t, s_est - 2 * np.sqrt(P_s), s_est + 2 * np.sqrt(P_s), color='blue', alpha=0.2)
    plt.title('Position')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid()

    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(t, v_true, label='True Velocity', color='black')
    plt.plot(t[1:], v_diff[1:], label='Finite Difference Velocity', color='orange')
    plt.plot(t, v_est, label='Estimated Velocity', color='blue')
    plt.fill_between(t, v_est - 2 * np.sqrt(P_v), v_est + 2 * np.sqrt(P_v), color='blue', alpha=0.2)
    plt.title('Velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid()

    # Acceleration plot
    plt.subplot(3, 1, 3)
    plt.plot(t, a_true, label='True Acceleration', color='black')
    plt.plot(t, a_est, label='Estimated Acceleration', color='blue')
    plt.fill_between(t, a_est - 2 * np.sqrt(P_a), a_est + 2 * np.sqrt(P_a), color='blue', alpha=0.2)
    plt.title('Acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_monte_carlo(t, results, bounds, n_runs, deceleration, start_up_steps):
    """
    Monte Carlo evaluation of the 3rd-order filter, one curve per process noise value Q.
    Args:
        t: np.ndarray, time vector
        results: dict Q -> (rmse_t, sigma_t, anees_t), each averaged across runs per time step
        bounds: (r1, r2), ANEES confidence interval for n = 3
        n_runs: int, number of Monte Carlo runs
        deceleration: dict with keys t_start_s, duration_s, v_end, to shade the braking phase
        start_up_steps: int, steps before this are excluded from the evaluation
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    (ax_s, ax_v), (ax_a, ax_n) = axs
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (Q, (rmse_t, sigma_t, anees_t)), color in zip(results.items(), colors):
        for ax, i in ((ax_s, 0), (ax_v, 1), (ax_a, 2)):
            ax.plot(t, rmse_t[:, i], color=color, label=f"RMSE, Var(z_a) = {Q:g}")
            ax.plot(t, sigma_t[:, i], color=color, ls="--", label=f"Filter σ, Var(z_a) = {Q:g}")
        ax_n.plot(t, anees_t, color=color, label=f"Var(z_a) = {Q:g}")

    r1, r2 = bounds
    ax_n.axhspan(r1, r2, color="gray", alpha=0.2, label=f"95% bounds [{r1:.2f}, {r2:.2f}]")
    ax_n.axhline(3, color="black", lw=1, ls=":")

    ax_s.set_ylabel("Position error (m)")
    ax_s.set_title("Position: RMSE vs. filter σ")
    ax_v.set_ylabel("Speed error (m/s)")
    ax_v.set_title("Speed: RMSE vs. filter σ")
    ax_a.set_ylabel("Acceleration error (m/s²)")
    ax_a.set_title("Acceleration: RMSE vs. filter σ")
    ax_n.set_ylabel("ANEES")
    ax_n.set_title("ANEES ([1], Eq. 9.12), target n = 3")
    ax_n.set_ylim(0, 10)   # the first steps and a too small Q can exceed this

    t_brake = (deceleration["t_start_s"], deceleration["t_start_s"] + deceleration["duration_s"])
    for ax in axs.flat:
        ax.axvspan(*t_brake, color="tab:orange", alpha=0.12, label="Braking")
        ax.axvline(t[start_up_steps], color="black", lw=0.8, ls="-.", label="End of start-up")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for ax in (ax_s, ax_v, ax_a):
        ax.set_yscale("log")
    for ax in (ax_a, ax_n):
        ax.set_xlabel("Time (s)")

    fig.suptitle(f"Stage 2: Monte Carlo evaluation of the 3rd-order filter, {n_runs} runs")
    fig.tight_layout()
    plt.show()
