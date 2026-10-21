import matplotlib.pyplot as plt
import numpy as np

AXES = ("x", "y")

def plot_simulation(t, s_true, v_true, a_true, s_measured, v_diff):
    """
    Plot the simulated ground truth and measured x-y-position data for the 2D surface radar scenario.
    Args:
        t: np.ndarray, time vector
        s_true: np.ndarray, true position vector (2D)
        v_true: np.ndarray, true velocity vector (2D)
        a_true: np.ndarray, true acceleration vector (2D)
        s_measured: np.ndarray, measured position vector (2D)
        v_diff: np.ndarray, estimated velocity vector using finite difference (2D)
    """

    plt.plot(s_true[:, 0], s_true[:, 1], label='True Position', color='blue')
    plt.scatter(s_measured[:, 0], s_measured[:, 1], label='Measured Position', color='red', s=10)
    plt.title('2D Surface Radar Simulation')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.legend()
    plt.axis('equal')

    plt.show()

def _plot_error(ax, t, raw_error, raw_label, kf_error, band, sensor_band, title, unit):
    """
    One error panel: raw error as points, filter error as a line, filter ±2σ as a band and the
    ±2σ of the raw signal as dashed lines.
    """
    ax.scatter(t, raw_error, label=raw_label, color="tab:red", s=8)
    ax.plot(t, kf_error, label="Kalman filter", color="tab:blue")
    ax.fill_between(t, -band, band, color="tab:blue", alpha=0.15, label="Filter ±2σ")
    ax.axhline(sensor_band, ls="--", color="tab:red", lw=1, label=f"{raw_label} ±2σ")
    ax.axhline(-sensor_band, ls="--", color="tab:red", lw=1)
    ax.set_ylim(-2 * sensor_band, 2 * sensor_band)  # first steps have a large band
    ax.set_title(title)
    ax.set_ylabel(f"Error ({unit})")

def _mark_maneuvers(ax, maneuvers):
    """
    Shade the time window of each maneuver, t_start_s to t_start_s + duration_s.
    """
    for i, m in enumerate(maneuvers):
        ax.axvspan(m["t_start_s"], m["t_start_s"] + m["duration_s"], color="gray", alpha=0.2,
                   label="Maneuver" if i == 0 else None)

def plot_all(t, s_true, s_measured, s_kf, v_true, v_diff, v_kf, P_kf, sigma_m, dt_s, maneuvers=()):
    """
    Results of one run of the 2D surface radar scenario.
    Top row: track, x and y position error. Bottom row: velocity, v_x and v_y error.
    Args:
        t: np.ndarray, time vector (n_steps,)
        s_true, s_measured, s_kf: np.ndarray, true / measured / filtered position (n_steps, 2)
        v_true, v_diff, v_kf: np.ndarray, true / finite difference / filtered velocity (n_steps, 2)
        P_kf: np.ndarray, corrected covariances P_tilde(k) for [x, y, v_x, v_y] (n_steps, 4, 4)
        sigma_m: float, measurement noise standard deviation per axis
        dt_s: float, time step in seconds
        maneuvers: list of dicts with keys t_start_s, duration_s, shaded in the time plots
    """
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    (ax_track, ax_ex, ax_ey), (ax_v, ax_evx, ax_evy) = axs

    # ±2σ of the filter from the diagonal of P_tilde, [x, y, v_x, v_y]
    band = 2 * np.sqrt(np.diagonal(P_kf, axis1=1, axis2=2))

    # Track
    ax_track.plot(s_true[:, 0], s_true[:, 1], label="Ground truth", color="black")
    ax_track.scatter(s_measured[:, 0], s_measured[:, 1], label="Measured", color="tab:red", s=8)
    ax_track.plot(s_kf[:, 0], s_kf[:, 1], label="Kalman filter", color="tab:blue")
    ax_track.plot(*s_true[0], "ko", label="Start")
    ax_track.set_xlabel("x (m)")
    ax_track.set_ylabel("y (m)")
    ax_track.set_title("Track")
    ax_track.set_aspect("equal", adjustable="datalim")

    # Position error per axis
    for ax, i in ((ax_ex, 0), (ax_ey, 1)):
        _plot_error(ax, t, s_measured[:, i] - s_true[:, i], "Measured", s_kf[:, i] - s_true[:, i],
                    band[:, i], 2 * sigma_m, f"Position error {AXES[i]}", "m")

    # Velocity components
    for i, color in ((0, "tab:blue"), (1, "tab:green")):
        ax_v.plot(t, v_true[:, i], color="black", ls="-" if i == 0 else "--",
                  label=f"Ground truth $v_{AXES[i]}$")
        ax_v.plot(t, v_kf[:, i], color=color, label=f"Kalman filter $v_{AXES[i]}$")
        ax_v.fill_between(t, v_kf[:, i] - band[:, 2 + i], v_kf[:, i] + band[:, 2 + i],
                          color=color, alpha=0.15)
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.set_title("Velocity (band: filter ±2σ)")
    ax_v.set_ylim(np.min(v_true) - 5, np.max(v_true) + 5)  # first steps have a large band

    # Velocity error per axis. A finite difference of two noisy positions has σ = √2·σ_m / dt.
    sigma_diff = np.sqrt(2) * sigma_m / dt_s
    for ax, i in ((ax_evx, 0), (ax_evy, 1)):
        _plot_error(ax, t, v_diff[:, i] - v_true[:, i], "Finite difference",
                    v_kf[:, i] - v_true[:, i], band[:, 2 + i], 2 * sigma_diff,
                    f"Velocity error $v_{AXES[i]}$", "m/s")
        ax.set_xlabel("Time (s)")

    for ax in (ax_ex, ax_ey, ax_v, ax_evx, ax_evy):
        _mark_maneuvers(ax, maneuvers)

    for ax in axs.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Stage 3: 2D surface radar, constant velocity model, position measurements")
    fig.tight_layout()
    plt.show()
